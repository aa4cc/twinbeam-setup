/**
 * @author  Martin Gurtner
 */

#ifndef BEADSTRACKER_H
#define BEADSTRACKER_H

#include <vector>
#include "Definitions.h"

class BeadTracker {
private:
    std::vector<Position> bead_positions;
    static std::vector<Position>::const_iterator closestBead(const std::vector<Position>& bp_all, const Position& bp_i);

public:
    BeadTracker();
    void update(const std::vector<Position>& bp_all);
    void addBead(const Position& bp);
    void deleteBead(const Position& bp);
    const std::vector<Position>& getBeadPositions();
    void clear();
};

#endif