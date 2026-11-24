/*
	Native rebuild for single-partition MXF recordings (prototype)

	Some recorders -- Canon XF-AVC is the case this was written against -- write
	OP1a files with no Body Partition: the Header Partition carries the header
	metadata, a pre-reserved index area, and then the essence itself
	(BodySID != 0 in the Header Partition Pack). While recording, the camera
	leaves durations at 0, the index segment empty, the start timecode at 0, and
	writes no Footer Partition or Random Index Pack. If recording is interrupted
	those are never filled in, and the clip has no timestamps or length.

	The existing reference-driven repair (mxf.cpp) splits files at the first
	Body Partition and then remuxes through FFmpeg's MXF muxer. Neither works
	here: there is no Body Partition, and FFmpeg's muxer corrupts long-GOP
	H.264 timing. This path instead rewrites the damaged file in place layout:

	  1. Parse both Header Partitions and require the same recorder profile:
	     identical KAG, byte counts, SIDs, Operational Pattern, Primer Pack, and
	     header metadata set layout.
	  2. Walk the damaged file's content packages from the essence start and
	     keep only complete ones. Element offsets must match the index segment's
	     DeltaEntryArray, so the package layout is known exactly.
	  3. Learn the index pattern (TemporalOffset, KeyFrameOffset, Flags) from
	     the reference, which must repeat with the GOP length. The reference's
	     own StreamOffsets and SliceOffsets must agree with its packages, and its
	     keyframe flags must agree with its H.264 IDR frames, or nothing is
	     written.
	  4. Check every kept damaged keyframe against H.264 IDR detection, then
	     trim trailing frames whose display order would reference a missing
	     frame.
	  5. Write the damaged header with Duration/ContainerDuration set to the
	     recovered length, StartTimecode restored from the first frame's
	     system item timecode, and the FooterPartition pointer set. Index
	     segments are written into the reserved index area so the essence stays
	     at its original offset. The essence is copied byte for byte, followed
	     by a Footer Partition and RIP modelled on the reference's.
	  6. Re-open the result with libavformat and check the packet count.

	The damaged clip keeps its own identity (UMIDs, instance UIDs, dates); the
	reference contributes only the index pattern and footer layout.
*/
#include "mxf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
}

using namespace std;

namespace {
typedef array<uint8_t, 16> UL;

struct Reader {
	ifstream f;
	uint64_t size = 0;
	string path;
	explicit Reader(const string& p) : f(p, ios::binary), path(p) {
		if (!f)
			throw runtime_error("could not open MXF file: " + p);
		f.seekg(0, ios::end);
		auto n = f.tellg();
		if (n < 0)
			throw runtime_error("could not size MXF file: " + p);
		size = uint64_t(n);
	}
	bool read(uint64_t off, void* dst, size_t n) {
		if (off > size || n > size - off)
			return false;
		f.clear();
		f.seekg(streamoff(off));
		return bool(f.read(reinterpret_cast<char*>(dst), streamsize(n)));
	}
	vector<uint8_t> bytes(uint64_t off, uint64_t n) {
		vector<uint8_t> v(static_cast<size_t>(n));
		if (!read(off, v.data(), size_t(n)))
			throw runtime_error("MXF read out of bounds in " + path);
		return v;
	}
};

uint64_t getBE(const uint8_t* p, int n) {
	uint64_t v = 0;
	for (int i = 0; i < n; ++i)
		v = (v << 8) | p[i];
	return v;
}
void putBE(uint8_t* p, uint64_t v, int n) {
	for (int i = n - 1; i >= 0; --i) {
		p[i] = uint8_t(v);
		v >>= 8;
	}
}
void appendBE(vector<uint8_t>& out, uint64_t v, int n) {
	size_t at = out.size();
	out.resize(at + size_t(n));
	putBE(out.data() + at, v, n);
}

struct Klv {
	uint64_t offset = 0, value = 0, length = 0;
	UL key{};
	uint64_t end() const { return value + length; }
};

// Reads a KLV header. Returns false for a non-SMPTE key, a bad BER length, or
// a header past EOF. The value may still extend past EOF; callers check end().
bool readKlv(Reader& r, uint64_t off, Klv& k) {
	if (off >= r.size || r.size - off < 17)
		return false;
	uint8_t h[25];
	size_t avail = size_t(min<uint64_t>(sizeof(h), r.size - off));
	if (!r.read(off, h, avail))
		return false;
	if (h[0] != 0x06 || h[1] != 0x0e || h[2] != 0x2b || h[3] != 0x34)
		return false;
	uint64_t len;
	size_t ber;
	if (h[16] < 0x80) {
		len = h[16];
		ber = 1;
	} else {
		size_t n = h[16] & 0x7f;
		if (!n || n > 8 || 17 + n > avail)
			return false;
		len = getBE(h + 17, int(n));
		ber = 1 + n;
	}
	copy(h, h + 16, k.key.begin());
	k.offset = off;
	k.value = off + 16 + ber;
	k.length = len;
	return true;
}

const uint8_t kFillTail[8] = {0x03, 0x01, 0x02, 0x10, 0x01, 0x00, 0x00, 0x00};
bool isFill(const UL& k) {
	return k[4] == 0x01 && k[5] == 0x01 && k[6] == 0x01 && equal(kFillTail, kFillTail + 8, k.begin() + 8);
}
const uint8_t kPartPrefix[13] = {0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01};
bool isPartitionFamily(const UL& k) { return equal(kPartPrefix, kPartPrefix + 13, k.begin()); }
bool isPartitionPack(const UL& k, uint8_t kind) { return isPartitionFamily(k) && k[13] == kind; }
bool isPrimer(const UL& k) { return isPartitionFamily(k) && k[13] == 0x05 && k[14] == 0x01; }
bool isRip(const UL& k) { return isPartitionFamily(k) && k[13] == 0x11; }
bool isIndexSegment(const UL& k) {
	static const uint8_t p[16] = {0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01,
		0x0d, 0x01, 0x02, 0x01, 0x01, 0x10, 0x01, 0x00};
	return equal(p, p + 16, k.begin());
}
// SMPTE 379 Generic Container essence element (picture, sound, data, ...).
bool isEssenceElement(const UL& k) {
	static const uint8_t p[12] = {0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01};
	return equal(p, p + 12, k.begin());
}

struct LocalItem {
	uint16_t tag = 0, length = 0;
	uint64_t value = 0;  // absolute offset of the value
};
vector<LocalItem> localItems(Reader& r, const Klv& k) {
	vector<uint8_t> v = r.bytes(k.value, k.length);
	vector<LocalItem> items;
	size_t p = 0;
	while (p + 4 <= v.size()) {
		LocalItem it;
		it.tag = uint16_t(getBE(&v[p], 2));
		it.length = uint16_t(getBE(&v[p + 2], 2));
		it.value = k.value + p + 4;
		if (p + 4 + it.length > v.size())
			throw runtime_error("MXF local set item overruns its set");
		items.push_back(it);
		p += 4 + it.length;
	}
	return items;
}

struct IndexSegment {
	Klv klv;
	vector<LocalItem> items;
	vector<uint8_t> raw;  // whole KLV value
	uint64_t duration = 0, startPosition = 0;
	uint32_t editUnitByteCount = 0, entryCount = 0, entrySize = 0;
	uint8_t sliceCount = 0, posTableCount = 0;
	vector<uint8_t> delta;  // DeltaEntryArray elements, 6 bytes each
	vector<uint8_t> entries;
};
IndexSegment parseIndexSegment(Reader& r, const Klv& k) {
	IndexSegment s;
	s.klv = k;
	s.items = localItems(r, k);
	s.raw = r.bytes(k.value, k.length);
	for (const auto& it : s.items) {
		const uint8_t* v = &s.raw[it.value - k.value];
		switch (it.tag) {
		case 0x3f0d: if (it.length == 8) s.duration = getBE(v, 8); break;
		case 0x3f0c: if (it.length == 8) s.startPosition = getBE(v, 8); break;
		case 0x3f05: if (it.length == 4) s.editUnitByteCount = uint32_t(getBE(v, 4)); break;
		case 0x3f08: if (it.length == 1) s.sliceCount = v[0]; break;
		case 0x3f0e: if (it.length == 1) s.posTableCount = v[0]; break;
		case 0x3f09:
			if (it.length >= 8 && getBE(v + 4, 4) == 6 && 8 + getBE(v, 4) * 6 == it.length)
				s.delta.assign(v + 8, v + it.length);
			else
				throw runtime_error("unsupported MXF DeltaEntryArray");
			break;
		case 0x3f0a:
			if (it.length < 8)
				throw runtime_error("malformed MXF IndexEntryArray");
			s.entryCount = uint32_t(getBE(v, 4));
			s.entrySize = uint32_t(getBE(v + 4, 4));
			if (8 + uint64_t(s.entryCount) * s.entrySize != it.length)
				throw runtime_error("MXF IndexEntryArray length mismatch");
			s.entries.assign(v + 8, v + it.length);
			break;
		}
	}
	return s;
}

struct Layout {
	Klv pack;
	vector<uint8_t> packValue;
	uint32_t kag = 0, indexSID = 0, bodySID = 0;
	uint64_t metaStart = 0, metaEnd = 0, essenceStart = 0;
	vector<Klv> sets;
	vector<IndexSegment> index;
};
Layout parseLayout(Reader& r, const char* label) {
	Layout L;
	if (!readKlv(r, 0, L.pack) || !isPartitionPack(L.pack.key, 2) || L.pack.length < 88 || L.pack.end() > r.size)
		throw runtime_error(string(label) + " MXF does not start with a valid Header Partition Pack");
	L.packValue = r.bytes(L.pack.value, L.pack.length);
	const uint8_t* v = L.packValue.data();
	L.kag = uint32_t(getBE(v + 4, 4));
	uint64_t headerBytes = getBE(v + 32, 8), indexBytes = getBE(v + 40, 8);
	L.indexSID = uint32_t(getBE(v + 48, 4));
	L.bodySID = uint32_t(getBE(v + 60, 4));
	if (!L.bodySID)
		throw runtime_error(string(label) + " MXF Header Partition carries no essence (BodySID 0); "
			"this layout is not supported");
	// HeaderByteCount is counted from the Primer Pack, after any KAG fill.
	uint64_t o = L.pack.end();
	Klv k;
	while (readKlv(r, o, k) && isFill(k.key))
		o = k.end();
	if (!readKlv(r, o, k) || !isPrimer(k.key))
		throw runtime_error(string(label) + " MXF header metadata does not start with a Primer Pack");
	L.metaStart = o;
	L.metaEnd = L.metaStart + headerBytes;
	L.essenceStart = L.metaEnd + indexBytes;
	if (L.essenceStart > r.size)
		throw runtime_error(string(label) + " MXF header byte counts exceed the file");
	for (o = L.metaStart; o < L.metaEnd; o = k.end()) {
		if (!readKlv(r, o, k) || k.end() > L.metaEnd)
			throw runtime_error(string(label) + " MXF header metadata is damaged");
		L.sets.push_back(k);
	}
	for (o = L.metaEnd; o < L.essenceStart; o = k.end()) {
		if (!readKlv(r, o, k) || k.end() > L.essenceStart)
			throw runtime_error(string(label) + " MXF index area is damaged");
		if (isIndexSegment(k.key))
			L.index.push_back(parseIndexSegment(r, k));
		else if (!isFill(k.key))
			throw runtime_error(string(label) + " MXF index area holds an unexpected KLV");
	}
	return L;
}

struct Package {
	uint64_t start = 0, end = 0;
	vector<uint64_t> elements;  // offsets relative to start
	uint64_t picture = 0, pictureLength = 0;  // absolute value offset/length of the picture element
	int pictures = 0;
};
struct Walk {
	vector<Package> packages;  // complete packages only
	uint64_t stop = 0;         // where the walk ended
	bool reachedFooter = false;
	size_t droppedIncomplete = 0;
};
// Walks content packages from `from`. A package starts at each KLV with the
// same key as the first one (the system item for XF-AVC). Its elements are
// the package-start KLV plus every Generic Container essence element.
Walk walkPackages(Reader& r, uint64_t from, size_t elementsPerPackage) {
	Walk w;
	Klv first;
	if (!readKlv(r, from, first))
		throw runtime_error("MXF essence does not start with a KLV");
	UL cpKey = first.key;
	Package cur;
	bool open = false;
	uint64_t o = from;
	Klv k;
	// A package is kept when every element arrived intact. Elements are only
	// recorded once their whole value is inside the file, so a cut inside an
	// element leaves the package short; a cut inside trailing fill only loses
	// padding, which is rewritten before the footer.
	auto close = [&](uint64_t end) {
		if (!open)
			return;
		cur.end = end;
		if (cur.elements.size() == elementsPerPackage)
			w.packages.push_back(cur);
		else
			++w.droppedIncomplete;
		open = false;
	};
	while (true) {
		if (!readKlv(r, o, k)) {
			close(o);
			break;
		}
		if (isPartitionFamily(k.key) && (k.key[13] == 3 || k.key[13] == 4 || k.key[13] == 0x11)) {
			if (k.key[13] == 3)
				throw runtime_error("MXF has a Body Partition inside header essence; layout not supported");
			close(o);
			w.reachedFooter = true;
			break;
		}
		if (k.end() > r.size) {
			close(o);
			break;
		}
		if (k.key == cpKey) {
			close(o);
			cur = Package();
			cur.start = o;
			cur.elements.push_back(0);
			open = true;
		} else if (!open) {
			throw runtime_error("MXF essence holds data before the first content package");
		} else if (isEssenceElement(k.key)) {
			cur.elements.push_back(o - cur.start);
			if (k.key[12] == 0x15) {
				cur.picture = k.value;
				cur.pictureLength = k.length;
				++cur.pictures;
			}
		} else if (!isFill(k.key) && k.key[4] != 0x02) {
			// System item packs and sets (0x02 registry) belong to the package start.
			throw runtime_error("MXF content package holds an unexpected KLV");
		}
		o = k.end();
	}
	w.stop = o;
	return w;
}

// Element layout must match the DeltaEntryArray. Returns slice start offsets
// (relative to package start) for slices 1..n.
vector<uint64_t> sliceStarts(const Package& p, const vector<uint8_t>& delta) {
	size_t n = delta.size() / 6;
	vector<uint64_t> starts;
	uint64_t sliceStart = 0;
	uint8_t slice = 0;
	for (size_t e = 0; e < n; ++e) {
		uint8_t s = delta[e * 6 + 1];
		uint32_t d = uint32_t(getBE(&delta[e * 6 + 2], 4));
		if (s != slice) {
			if (s != slice + 1)
				throw runtime_error("unsupported MXF slice order");
			slice = s;
			sliceStart = p.elements[e];
			starts.push_back(sliceStart);
		}
		if (p.elements[e] - sliceStart != d)
			throw runtime_error("MXF content package layout does not match its index DeltaEntryArray");
	}
	return starts;
}

// H.264 Annex B: true for an IDR picture, false for a non-IDR picture.
int h264IsIdr(Reader& r, const Package& p) {
	for (uint64_t want : {uint64_t(4096), uint64_t(65536)}) {
		uint64_t n = min(want, p.pictureLength);
		vector<uint8_t> b = r.bytes(p.picture, n);
		for (size_t i = 0; i + 3 < b.size(); ++i)
			if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1) {
				uint8_t t = b[i + 3] & 0x1f;
				if (t == 5)
					return 1;
				if (t == 1)
					return 0;
			}
		if (n == p.pictureLength)
			break;
	}
	return -1;
}

// SMPTE 12M timecode from the user date/time stamp of an SMPTE 385 system
// metadata pack. For rates above 30 the frame pair flag is bit 7 of the
// seconds byte. Returns -1 when no timecode is present.
int64_t systemItemTimecode(Reader& r, const Package& p, int base, bool drop) {
	Klv k;
	if (!readKlv(r, p.start, k) || k.length < 45)
		return -1;
	vector<uint8_t> v = r.bytes(k.value, 45);
	if (v[40] != 0x81)
		return -1;
	auto bcd = [](uint8_t x, uint8_t mask) { return int(((x & mask) >> 4) * 10 + (x & 0x0f)); };
	int fr = bcd(v[41], 0x3f), sec = bcd(v[42], 0x7f), min = bcd(v[43], 0x7f), hr = bcd(v[44], 0x3f);
	if (base > 30)
		fr = fr * 2 + (v[42] >> 7);
	int64_t total = ((int64_t(hr) * 60 + min) * 60 + sec) * base + fr;
	if (drop) {
		int64_t minutes = int64_t(hr) * 60 + min;
		total -= (base / 30) * 2 * (minutes - minutes / 10);
	}
	return total;
}

struct Pattern {
	uint32_t period = 0;
	vector<array<uint8_t, 3>> ref;  // TemporalOffset, KeyFrameOffset, Flags per reference entry
	array<uint8_t, 3> at(uint64_t j) const {
		return j < period ? ref[size_t(j)] : ref[size_t(period + (j - period) % period)];
	}
};

void writeAll(ofstream& out, const uint8_t* p, size_t n) {
	out.write(reinterpret_cast<const char*>(p), streamsize(n));
	if (!out)
		throw runtime_error("could not write repaired MXF");
}
void copyBytes(Reader& in, ofstream& out, uint64_t begin, uint64_t end) {
	vector<uint8_t> b(4 << 20);
	while (begin < end) {
		size_t n = size_t(min<uint64_t>(b.size(), end - begin));
		if (!in.read(begin, b.data(), n))
			throw runtime_error("damaged MXF ended while copying essence");
		writeAll(out, b.data(), n);
		begin += n;
	}
}
vector<uint8_t> fillKlv(const UL& key, uint64_t total) {
	if (total == 0)
		return {};
	vector<uint8_t> f(key.begin(), key.end());
	if (total >= 20) {
		f.push_back(0x83);
		appendBE(f, total - 20, 3);
	} else if (total >= 17) {
		f.push_back(uint8_t(total - 17));
	} else {
		throw runtime_error("no room for an MXF fill item");
	}
	f.resize(size_t(total), 0);
	return f;
}

int64_t countPackets(const string& path, int& videoStream) {
	AVFormatContext* c = nullptr;
	if (avformat_open_input(&c, path.c_str(), nullptr, nullptr) < 0)
		return -1;
	if (avformat_find_stream_info(c, nullptr) < 0) {
		avformat_close_input(&c);
		return -1;
	}
	videoStream = av_find_best_stream(c, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
	AVPacket* pkt = av_packet_alloc();
	int64_t n = 0;
	while (av_read_frame(c, pkt) >= 0) {
		if (pkt->stream_index == videoStream)
			++n;
		av_packet_unref(pkt);
	}
	av_packet_free(&pkt);
	avformat_close_input(&c);
	return n;
}
} // namespace

bool MxfRepair::isSinglePartition(const std::string& reference) {
	Reader r(reference);
	Klv k;
	if (!readKlv(r, 0, k) || !isPartitionPack(k.key, 2) || k.length < 88)
		return false;
	vector<uint8_t> v = r.bytes(k.value, 88);
	return getBE(&v[60], 4) != 0;
}

void MxfRepair::rebuildSinglePartition(const string& reference, const string& damaged, const string& destination) {
	Reader R(reference), D(damaged);
	Layout lr = parseLayout(R, "reference"), ld = parseLayout(D, "damaged");

	// --- 1. Same recorder profile -------------------------------------------
	auto same = [](const vector<uint8_t>& a, const vector<uint8_t>& b, size_t from, size_t to) {
		return equal(a.begin() + from, a.begin() + to, b.begin() + from);
	};
	if (lr.kag != ld.kag || lr.indexSID != ld.indexSID || lr.bodySID != ld.bodySID ||
		lr.metaEnd - lr.metaStart != ld.metaEnd - ld.metaStart ||
		lr.essenceStart - lr.metaEnd != ld.essenceStart - ld.metaEnd ||
		!same(lr.packValue, ld.packValue, 64, 80))
		throw runtime_error("reference and damaged MXF Header Partitions differ (KAG, SIDs, sizes, or Operational Pattern)");
	if (lr.sets.size() != ld.sets.size())
		throw runtime_error("reference and damaged MXF header metadata differ in layout");
	for (size_t i = 0; i < lr.sets.size(); ++i)
		if (lr.sets[i].key != ld.sets[i].key || lr.sets[i].length != ld.sets[i].length)
			throw runtime_error("reference and damaged MXF header metadata differ in layout");
	if (R.bytes(lr.sets[0].offset, lr.sets[0].end() - lr.sets[0].offset) !=
		D.bytes(ld.sets[0].offset, ld.sets[0].end() - ld.sets[0].offset))
		throw runtime_error("reference and damaged MXF Primer Packs differ");
	if (lr.index.size() != 1 || ld.index.size() != 1)
		throw runtime_error("expected exactly one index segment in each Header Partition");
	const IndexSegment& ir = lr.index[0];
	const IndexSegment& id = ld.index[0];
	if (ir.editUnitByteCount || ir.posTableCount || id.posTableCount || ir.delta != id.delta ||
		ir.sliceCount != id.sliceCount || ir.delta.empty() || ir.entrySize != 11 + 4u * ir.sliceCount ||
		ir.startPosition != 0 || ir.entryCount != ir.duration)
		throw runtime_error("unsupported or mismatched MXF index segment");
	size_t perPackage = ir.delta.size() / 6;

	// --- 2. Content packages ------------------------------------------------
	Walk wr = walkPackages(R, lr.essenceStart, perPackage);
	Walk wd = walkPackages(D, ld.essenceStart, perPackage);
	if (!wr.reachedFooter || wr.droppedIncomplete || wr.packages.size() != ir.entryCount)
		throw runtime_error("reference MXF is not a complete recording (" + to_string(wr.packages.size()) +
			" packages, index has " + to_string(ir.entryCount) + ")");
	if (wd.packages.empty())
		throw runtime_error("damaged MXF has no complete content packages");
	if (wr.packages[0].start != lr.essenceStart || wd.packages[0].start != ld.essenceStart)
		throw runtime_error("MXF essence does not start at the indexed position");

	// --- 3. Learn and validate the index pattern on the reference -----------
	Pattern pat;
	vector<uint32_t> keyframes;
	for (uint32_t i = 0; i < ir.entryCount; ++i) {
		const uint8_t* e = &ir.entries[size_t(i) * ir.entrySize];
		const Package& p = wr.packages[i];
		if (getBE(e + 3, 8) != p.start - lr.essenceStart)
			throw runtime_error("reference index StreamOffset does not match its content packages");
		vector<uint64_t> slices = sliceStarts(p, ir.delta);
		if (slices.size() != ir.sliceCount)
			throw runtime_error("reference slice count does not match its DeltaEntryArray");
		for (size_t s = 0; s < slices.size(); ++s)
			if (getBE(e + 11 + 4 * s, 4) != slices[s])
				throw runtime_error("reference index SliceOffset does not match its content packages");
		if (p.pictures != 1)
			throw runtime_error("expected one picture element per content package");
		int idr = h264IsIdr(R, p);
		if (idr < 0)
			throw runtime_error("reference picture essence is not H.264 Annex B (only H.264 is supported)");
		if (bool(e[2] & 0x80) != bool(idr))
			throw runtime_error("reference keyframe flags do not match its H.264 IDR frames");
		if (idr)
			keyframes.push_back(i);
		pat.ref.push_back({e[0], e[1], e[2]});
	}
	if (keyframes.size() < 3 || keyframes[0] != 0)
		throw runtime_error("reference MXF is too short to learn its GOP pattern (needs 2 full GOPs)");
	pat.period = keyframes[1] - keyframes[0];
	for (size_t i = 2; i < keyframes.size(); ++i)
		if (keyframes[i] - keyframes[i - 1] != pat.period)
			throw runtime_error("reference GOP length is not constant");
	// The last reference GOP may end early; only full GOPs are compared.
	uint32_t fullEnd = keyframes.back();
	for (uint32_t j = pat.period; j < fullEnd; ++j)
		if (pat.ref[j] != pat.ref[j - pat.period])
			throw runtime_error("reference index pattern does not repeat every GOP");

	// --- 4. Validate the damaged packages and trim the tail ------------------
	uint64_t n = wd.packages.size();
	for (uint64_t j = 0; j < n; ++j) {
		const Package& p = wd.packages[size_t(j)];
		sliceStarts(p, id.delta);  // throws if the layout differs
		if (p.pictures != 1)
			throw runtime_error("damaged MXF content package " + to_string(j) + " has no single picture element");
		int idr = h264IsIdr(D, p);
		bool key = pat.at(j)[2] & 0x80;
		if (idr < 0 || bool(idr) != key)
			throw runtime_error("damaged MXF frame " + to_string(j) + (key ? " should be" : " should not be") +
				" a keyframe according to the reference GOP pattern; recorder settings differ");
	}
	uint64_t complete = n;
	auto displayOk = [&](uint64_t count) {
		for (uint64_t d = (count > 2 * pat.period ? count - 2 * pat.period : 0); d < count; ++d) {
			int64_t stored = int64_t(d) + int8_t(pat.at(d)[0]);
			if (stored < 0 || uint64_t(stored) >= count)
				return false;
		}
		return true;
	};
	while (n > 0 && !displayOk(n))
		--n;
	if (!n)
		throw runtime_error("no displayable MXF frames could be recovered");

	// --- 5a. Header metadata patches -----------------------------------------
	vector<uint8_t> header = D.bytes(0, ld.metaEnd);
	uint64_t refFrames = ir.entryCount;

	// Timecode base and drop flag from the TimecodeComponent (06.0e.2b.34.02.53.01.01.0d.01.01.01.01.01.14.00).
	int tcBase = 0;
	bool tcDrop = false;
	for (const auto& s : ld.sets)
		if (s.key[4] == 0x02 && s.key[5] == 0x53 && s.key[13] == 0x01 && s.key[14] == 0x14)
			for (const auto& it : localItems(D, s)) {
				if (it.tag == 0x1502 && it.length == 2)
					tcBase = int(getBE(&header[size_t(it.value)], 2));
				if (it.tag == 0x1503 && it.length == 1)
					tcDrop = header[size_t(it.value)] != 0;
			}
	int64_t refTc = -1, newTc = -1;
	if (tcBase > 0) {
		refTc = systemItemTimecode(R, wr.packages[0], tcBase, tcDrop);
		newTc = systemItemTimecode(D, wd.packages[0], tcBase, tcDrop);
	}
	bool tcValidated = false;

	for (size_t i = 0; i < ld.sets.size(); ++i) {
		// Only 2-byte-tag local sets carry these properties; the Primer Pack and
		// fill items do not parse as local sets and are skipped.
		if (isPrimer(ld.sets[i].key) || isFill(ld.sets[i].key))
			continue;
		vector<LocalItem> ri, di;
		try {
			ri = localItems(R, lr.sets[i]);
			di = localItems(D, ld.sets[i]);
		} catch (const runtime_error&) {
			continue;
		}
		if (ri.size() != di.size())
			continue;
		for (size_t t = 0; t < ri.size(); ++t) {
			if (ri[t].tag != di[t].tag || ri[t].length != 8)
				continue;
			uint16_t tag = ri[t].tag;
			if (tag != 0x0202 && tag != 0x3002 && tag != 0x1501)
				continue;
			uint64_t rv = getBE(R.bytes(ri[t].value, 8).data(), 8);
			uint8_t* dp = &header[size_t(di[t].value)];
			uint64_t dv = getBE(dp, 8);
			if (tag == 0x1501) {
				if (refTc >= 0 && rv == uint64_t(refTc)) {
					tcValidated = true;
					if (newTc >= 0)
						putBE(dp, uint64_t(newTc), 8);
				}
			} else if (rv == refFrames) {
				putBE(dp, n, 8);
			} else if (rv != dv) {
				throw runtime_error("unexpected MXF duration field; not patched");
			}
		}
	}

	// --- 5b. Index segments into the reserved area ---------------------------
	const uint64_t maxPerSegment = (65535 - 8) / id.entrySize;
	vector<uint8_t> indexArea;
	for (uint64_t first = 0, seg = 0; first < n; first += maxPerSegment, ++seg) {
		uint64_t count = min(maxPerSegment, n - first);
		vector<uint8_t> value;
		for (const auto& it : id.items) {
			const uint8_t* src = &id.raw[size_t(it.value - id.klv.value)];
			appendBE(value, it.tag, 2);
			if (it.tag == 0x3f0a) {
				appendBE(value, 8 + count * id.entrySize, 2);
				appendBE(value, count, 4);
				appendBE(value, id.entrySize, 4);
				for (uint64_t j = first; j < first + count; ++j) {
					const Package& p = wd.packages[size_t(j)];
					auto e = pat.at(j);
					value.insert(value.end(), e.begin(), e.end());
					appendBE(value, p.start - ld.essenceStart, 8);
					for (uint64_t s : sliceStarts(p, id.delta))
						appendBE(value, s, 4);
				}
				continue;
			}
			appendBE(value, it.length, 2);
			size_t at = value.size();
			value.insert(value.end(), src, src + it.length);
			if (it.tag == 0x3f0d && it.length == 8)
				putBE(&value[at], count, 8);
			else if (it.tag == 0x3f0c && it.length == 8)
				putBE(&value[at], first, 8);
			else if (it.tag == 0x3c0a && it.length == 16)
				value[at + 15] = uint8_t(value[at + 15] + seg);  // unique InstanceUID per segment
		}
		if (find_if(id.items.begin(), id.items.end(), [](const LocalItem& i) { return i.tag == 0x3f0a; }) == id.items.end())
			throw runtime_error("damaged MXF index segment has no IndexEntryArray to fill");
		indexArea.insert(indexArea.end(), id.klv.key.begin(), id.klv.key.end());
		indexArea.push_back(0x83);
		appendBE(indexArea, value.size(), 3);
		indexArea.insert(indexArea.end(), value.begin(), value.end());
	}
	uint64_t reserved = ld.essenceStart - ld.metaEnd;
	if (indexArea.size() + 20 > reserved)
		throw runtime_error("recovered index (" + to_string(n) + " frames) does not fit the reserved index area; "
			"clips this long are not supported by the prototype yet");
	UL fillKey{};
	{
		Klv k;
		for (uint64_t o = ld.metaEnd; o < ld.essenceStart && readKlv(D, o, k); o = k.end())
			if (isFill(k.key)) {
				fillKey = k.key;
				break;
			}
		if (!isFill(fillKey))
			throw runtime_error("could not find an MXF fill key to reuse");
	}
	vector<uint8_t> pad = fillKlv(fillKey, reserved - indexArea.size());
	indexArea.insert(indexArea.end(), pad.begin(), pad.end());

	// --- 5c. Footer and RIP modelled on the reference ------------------------
	uint64_t essenceEnd = wd.packages[size_t(n - 1)].end;
	uint64_t kag = ld.kag ? ld.kag : 1;
	uint64_t footer = (essenceEnd + kag - 1) / kag * kag;
	if (footer != essenceEnd && footer - essenceEnd < 17)
		footer += kag;
	vector<uint8_t> gap = fillKlv(fillKey, footer - essenceEnd);

	uint64_t refFooter = wr.stop;
	vector<uint8_t> tail = R.bytes(refFooter, R.size - refFooter);
	{
		Klv fk;
		if (!readKlv(R, refFooter, fk) || !isPartitionPack(fk.key, 4) || fk.length < 88)
			throw runtime_error("reference MXF has no Footer Partition Pack");
		size_t v = size_t(fk.value - refFooter);
		if (getBE(&tail[v + 16], 8) != 0 || getBE(&tail[v + 32], 8) || getBE(&tail[v + 40], 8))
			throw runtime_error("reference Footer Partition carries metadata or index; not supported");
		putBE(&tail[v + 8], footer, 8);
		putBE(&tail[v + 16], 0, 8);
		putBE(&tail[v + 24], footer, 8);
		bool rip = false;
		for (uint64_t o = refFooter; o < R.size; o = fk.end()) {
			if (!readKlv(R, o, fk) || fk.end() > R.size)
				throw runtime_error("reference MXF footer is malformed");
			if (!isRip(fk.key))
				continue;
			rip = true;
			if (fk.length < 4 || (fk.length - 4) % 12)
				throw runtime_error("reference MXF RIP is malformed");
			for (uint64_t p = fk.value - refFooter; p + 12 <= fk.end() - 4 - refFooter; p += 12) {
				uint64_t off = getBE(&tail[size_t(p + 4)], 8);
				if (off == refFooter)
					putBE(&tail[size_t(p + 4)], footer, 8);
				else if (off != 0)
					throw runtime_error("reference MXF RIP lists partitions this rebuild cannot place");
			}
		}
		if (!rip)
			throw runtime_error("reference MXF has no Random Index Pack");
	}
	putBE(&header[size_t(ld.pack.value + 24)], footer, 8);
	header[14] = R.bytes(0, 16)[14];  // partition status as in the finished reference

	// --- 5d. Write ------------------------------------------------------------
	string target = destination.empty() ? damaged + "_fixed.mxf" : destination;
	string partial = target + ".untrunc-partial";
	try {
		{
			ofstream out(partial, ios::binary | ios::trunc);
			if (!out)
				throw runtime_error("could not create repaired MXF: " + partial);
			writeAll(out, header.data(), header.size());
			writeAll(out, indexArea.data(), indexArea.size());
			copyBytes(D, out, ld.essenceStart, essenceEnd);
			writeAll(out, gap.data(), gap.size());
			writeAll(out, tail.data(), tail.size());
			out.close();
			if (!out)
				throw runtime_error("could not finish writing repaired MXF");
		}
		// --- 6. Verify with libavformat --------------------------------------
		int vs = -1;
		int64_t packets = countPackets(partial, vs);
		if (packets != int64_t(n))
			throw runtime_error("rebuilt MXF verification failed: libavformat reads " + to_string(packets) +
				" video packets, expected " + to_string(n));
		if (rename(partial.c_str(), target.c_str()))
			throw runtime_error("could not move repaired MXF into place: " + target);
	} catch (...) {
		remove(partial.c_str());
		throw;
	}

	cout << "Rebuilt single-partition MXF: " << n << " frames recovered";
	if (complete > n)
		cout << " (" << complete - n << " trailing frame(s) dropped: their display order needs missing frames)";
	if (wd.droppedIncomplete)
		cout << ", " << wd.droppedIncomplete << " incomplete content package(s) dropped";
	cout << "\n  GOP pattern learned from reference: " << pat.period << " frames";
	if (tcValidated && newTc >= 0)
		cout << "\n  start timecode restored from the first frame: " << newTc << " frames @" << tcBase
			 << (tcDrop ? " drop-frame" : "");
	else
		cout << "\n  start timecode left unchanged (could not be validated against the reference)";
	cout << "\n  output: " << target << endl;
}
