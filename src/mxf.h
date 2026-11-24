#ifndef MXF_H
#define MXF_H

#include <string>

class MxfRepair {
public:
    static void repair(const std::string& reference, const std::string& damaged,
                       const std::string& destination = std::string());
    // Single-partition recordings (essence in the Header Partition, e.g. Canon
    // XF-AVC): rebuild index, durations, timecode, footer, and RIP natively.
    static bool isSinglePartition(const std::string& reference);
    static void rebuildSinglePartition(const std::string& reference, const std::string& damaged,
                                       const std::string& destination);
};

#endif