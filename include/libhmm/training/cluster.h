#ifndef CLUSTER_H_
#define CLUSTER_H_

#include <vector>
#include <algorithm>
#include "libhmm/training/centroid.h"

namespace libhmm
{

/*
 * Represents a cluster of values "centered" around a Centroid.  Used in a
 * k-means clustering algorithm.
 */
class Cluster
{
    std::vector<Observation> observations;
    Centroid centroid;
public:
    Cluster( ){ }    

    /*
     * Creates a new Cluster based on one Observation.
     */
    Cluster( Observation o ){
        centroid.add( o, observations.size( ) );
        observations.push_back( o );
    }

    /*
     * Adds an observation to the Cluster.
     */
    void onlineAdd( Observation o ){
        centroid.add( o, observations.size( ) );
        observations.push_back( o );
    }

    /*
     * Adds an observation to the Cluster *without* recomputing the Centroid.
     * This is needed so that the Centroid does not shift during the initial
     * clustering phase to become abnormally large.
     */
    void batchAdd( Observation o ){
        observations.push_back( o );
    }

    /*
     * Recomputes the Centroid.
     */
    void recalculateCentroid( ){
        centroid.setValue( observations );
    }

    /* 
     * Removes an Observation from the cluster.
     *
     * This is kinda interesting...we need to know *where* in the Vector
     * our Observation is, but the ViterbiTrainer (the only class which will
     * be calling this function) should have no frickin' clue where to look.
     *
     * I suppose we could get away with removing the first instance of the
     * particular Observation, since the Centroid will still have to be
     * reevaluated without the Observation.  In that case, even if there
     * were duplicates in the list, mathematically it should have no 
     * effect on the Centroid.  I hope.
     */
    void remove( Observation o ){
        auto i = std::find(observations.begin(), observations.end(), o);
        
        // If observation is found, remove it
        if (i != observations.end()) {
            centroid.remove( o, observations.size( ) );
            observations.erase( i );
        }
        // Note: If observation not found, we silently ignore it to handle
        // edge cases with NaN values or numerical precision issues
    }

    /*
     * Returns the observations associated with the Cluster
     */
    std::vector<Observation> getObservations( ){
        return observations;
    }

    /*
     * Returns the distance from the Cluster's Centroid.
     */
    double getDistance( Observation o ) const {
        return centroid.distance( o );
    }

    /*
     * Returns the number of Observations associated with the Cluster.
     */
    int size( ){
        return observations.size( );
    }
    
    /*
     * Returns the centroid's value.
     */
    double getCentroidValue( ){
        return centroid.getValue( );
    }

    void init( Observation o ){
        centroid.setValue( o );
    }
}; // class Cluster

} //namespace

#endif
