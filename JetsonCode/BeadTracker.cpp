#include "BeadTracker.h"

using namespace std;

BeadTracker::BeadTracker() {
    // Allocate the memory for the bead_position array so that the dynamic memory allocation is avoided
    bead_positions.reserve(MAX_NUMBER_BEADS);
}

void BeadTracker::update(const vector<Position> &bp_all) {
    for(auto &b: bead_positions) {
        auto b_clst = BeadTracker::closestBead(bp_all, b);
        b.x = b_clst->x;
        b.y = b_clst->y;
    }
}

void BeadTracker::addBead(const Position& bp) {
    bead_positions.push_back(bp);
}

void BeadTracker::deleteBead(const Position& bp) {
    // Find the closest bead to the given one and remove it from the array of tracked beads
    bead_positions.erase(BeadTracker::closestBead(bead_positions, bp));
}

void BeadTracker::clear() {
    bead_positions.clear();
}

const vector<Position>& BeadTracker::getBeadPositions() {
    return bead_positions;
}

vector<Position>::const_iterator BeadTracker::closestBead(const vector<Position>& bp_all, const Position& bp_i) {
    int min_dist = -1;
    auto min_b = bp_all.begin();
    for(auto b=bp_all.begin(); b!=bp_all.end(); ++b) {
        int dist = ((int)b->x - (int)bp_i.x)*((int)b->x - (int)bp_i.x) + ((int)b->y - (int)bp_i.y)*((int)b->y - (int)bp_i.y);
        if(dist < min_dist) {
            min_dist = dist;
            min_b = b;
        }
    }

    return min_b;
}