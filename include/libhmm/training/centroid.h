#ifndef CENTROID_H_
#define CENTROID_H_

#include "libhmm/common/common.h"
#include <cmath>

namespace libhmm{
/*
 * Represents the centroid of a set of Observations.
 *
 * Based somewhat on the CentroidObservationInteger and Centroid classes in 
 * JAHMM. Modified to remove the generics but copied the comments.  Oh yeah, 
 * and *much* less heap memory usage.
 *
 * --Todd Jackson, Royal Military College of Canada
 */
class Centroid
{
    double value;    
public:

    Centroid( ){
        value = 0;
    }

    /*
     * Centroid based on one value.
     *
     * The average of one value is the value, right?
     */
    Centroid( Observation o ){
        value = o;
    }

    /**
     * Recompute the value of this centroid due to the addition of an
     * Observation to the cluster.
     *
     * The centroid will now be: 
     *
     *      sum( observations ) + o 
     *    ---------------------------
     *      num( observations ) + 1
     *  
     *  but since we have the value of the centroid already, we know that
     *
     *      sum( observations ) = num( observations ) * value
     *
     *  therefore, the centroid is now
     *
     *     num( observations ) * value + o 
     *    ---------------------------------
     *         num( observations ) + 1
     *
     */    
    void add( Observation o, int size){
        value = ( ( value * size ) + o ) / ( size + 1 );
    }

    /*
     * Recompute the value of this centroid due to the removal of an Observation
     * from a cluster.
     *
     * The centroid will now be: 
     *
     *      sum( observations ) - o 
     *    ---------------------------
     *      num( observations ) - 1
     *  
     *  but since we have the value of the centroid already, we know that
     *
     *      sum( observations ) = num( observations ) * value
     *
     *  therefore, the centroid is now
     *
     *     num( observations ) * value - o 
     *    ---------------------------------
     *         num( observations ) - 1
     *
     */  
    void remove( Observation o, int size ){
        value = ( ( value * size ) - o ) 
                  / ( size - 1 );
    }

    /*
     * Returns the Euclidean (aka linear) distance from the centroid to the
     * Observation
	 */
    double distance( Observation o ) const {
        return fabs( o - value );
    }

    /*
     * Forces a recalculation of the Centroid based on the given list of
     * Observations.
     */
    void setValue( std::vector<Observation> observations ){
        unsigned int i;
        double sum = 0;
        for( i = 0; i < observations.size( ); i++ ){
            sum += observations[ i ];
        }

        value = sum / observations.size( );
    }
    
    bool isNull( ){
        if( value == -1 ) return true;
        else return false;
    }

    double getValue( ){
        return value;
    }

    /*
     * Forces the Centroid to take a particular value.
     */
    void setValue( double _value ){
        value = _value;
    }
};

}

#endif
