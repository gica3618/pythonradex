/*
 *  model.c
 *  This file is part of LIME, the versatile line modeling engine
 *
 *  Copyright (C) 2006-2014 Christian Brinch
 *  Copyright (C) 2015-2017 The LIME development team
 *
 */

#include "lime.h"
#include "grid_values.h"

/******************************************************************************/

/*Interpolation function*/

double grid_interpolation_3D(double x, double y, double z, double values[grid_x_size][grid_y_size][grid_z_size], double floor_value){
    int ir,ix,iy,iz;
    double result;
    if ((x<grid_x[0]) || (x>grid_x[grid_x_size-1]) || (y<grid_y[0]) || (y>grid_y[grid_y_size-1]) || (z<grid_z[0]) || (z>grid_z[grid_z_size-1])){
        result = floor_value;
    }
    else {
        ix = gsl_interp_bsearch(grid_x, x, 0, grid_x_size-1);
        iy = gsl_interp_bsearch(grid_y, y, 0, grid_y_size-1);
        iz = gsl_interp_bsearch(grid_z, z, 0, grid_z_size-1);
        result = values[ix][iy][iz];
    }
    return result;
}
/********************************************************************************/

void
input(inputPars *par, image *img){
  int i;

  /*
   * Basic parameters. See cheat sheet for details.
   */
  par->radius                   = 300*AU;
  par->minScale                 = 0.5*AU;
  par->pIntensity               = 7000;
  par->sinkPoints               = 5000;
  //par->dust                     = "jena_thin_e6.tab";
  par->moldatfile[0]            = "../../tests/LAMDA_files/c.dat";
  par->sampling                 = 2; // log distr. for radius, directions distr. uniformly on a sphere.
  par->nSolveIters              = 14;
  par->nThreads = 10;
  par->lte_only = 0;
  //par->resetRNG	                = 0;

  //par->outputfile               = "populations.pop";
  //par->binoutputfile            = "restart.pop";
  //par->gridfile                 = "grid.vtk";

  /*
    Setting elements of the following three arrays is optional. NOTE
    that, if you do set any of their values, you should set as many as
    the number of elements returned by your function density(). The
    ith element of the array in question will then be assumed to refer
    to the ith element in the density function return. The current
    maximum number of elements allowed is 7, which is the number of
    types of collision partner recognized in the LAMBDA database.

    Note that there is no (longer) a hard connection between the
    number of density elements and the number of collision-partner
    species named in the moldata files. This means in practice that,
    if you set the values for par->collPartIds, you can, if you like,
    set some for which there are no transition rates supplied in the
    moldatfiles. This might happen for example if there is a molecule
    which contributes significantly to the total molecular density but
    for which there are no measured collision rates for the radiating
    species you are interested in.

    You may also omit to mention in par->collPartIds a collision
    partner which is specified in the moldatfiles. In this case LIME
    will assume the density of the respective molecules is zero.

    If you don't set any values for any or all of these arrays,
    i.e. if you omit any mention of them here (we preserve this
    possibility for purposes of backward compatibility), LIME will
    attempt to replicate the algorithms employed in version 1.5, which
    involve guessing which collision partner corresponds to which
    density element. Since this was not exactly a rigorous procedure,
    we recommend use of the arrays.

    par->nMolWeights: this specifies how you want the number density
    of each radiating species to be calculated. At each grid point a
    sum (weighted by par->nMolWeights) of the density values is made,
    then this is multiplied by the abundance to return the number
    density.

    par->collPartNames: this helps make a firm connection between the density functions and the collision partner information in the moldatfile.

    par->collPartMolWeights: this now allows control over the calculation of the dust opacity.

    Note that there are convenient macros defined in ../src/collparts.h for
    7 types of collision partner.

    Below is an example of how you might use these parameters:
  */

  par->collPartIds[0]           = CP_e;
  //par->nMolWeights[0]           = 1.0;
 // par->collPartNames[0]         = "e";
//  par->collPartMolWeights[0]    = 2.0159;

  /* Set one or more of the following parameters for full output of the grid-specific data at any of 4 stages during the processing. (See the header of gridio.c for information about the stages.)
  par->gridOutFiles[0] = "grid_1.ds";
  par->gridOutFiles[1] = "grid_2.ds";
  par->gridOutFiles[2] = "grid_3.ds";
  par->gridOutFiles[3] = "grid_4.ds";
  par->gridOutFiles[4] = "grid_5.ds";
  */

//par->gridOutFiles[4] = "grid_5.ds";


  /* You can also optionally read in a FITS file stored via the previous parameters, or prepared externally. See the header of grid2fits.c for information about the correct file format. LIME can cope with almost any sensible subset of the recognized columns; it will use the file values if they are present, then calculate the missing ones.
  par->gridInFile = "grid_5.ds";
  */

  /*
   * Definitions for image #0. Add blocks with successive values of i for additional images.
   */
  float PI=3.1415926535897932384;
  //img[i].nchan                  = 201;             // Number of channels
  //img[i].velres                 = 297.4;           // Channel resolution in m/s
  //img[i].trans                  = 0;              // zero-indexed J quantum number
  //img[i].pxls                   = 200;            // Pixels per dimension
  //img[i].imgres                 = 0.1;            // Resolution in arc seconds
  //img[i].distance               = 19.4*PC;         // source distance in m
  //img[i].source_vel             = 0;              // source velocity in m/s
  //img[i].theta = 90./180.*PI;
  //img[i].phi = 0.0;
  //img[i].azimuth                = 0.0;
  //img[i].incl                   = 0.0; //
  //img[i].posang			= 90./180.*PI;	

  /* For each set of image parameters above, numerous images with different units can be outputted. This is done by
   * setting img[].units to a delimited (space, comma, colon, underscore) string of the required outputs, where:
   *        0:Kelvin
   *        1:Jansky/pixel
   *        2:SI
   *        3:Lsun/pixel
   *        4:tau
   * If multiple units are specified for a single set of image parameters (e.g. "0 1 2") then the unit name is added
   * automatically at the end of the given filename, but before the filename extension if it exists. Otherwise if a
   * single unit is specified then the filename is unchanged.

   * A single image unit can also be specified for each image using img[].unit as in previous LIME versions. Note that
   * only img[].units or img[].unit should be set for each image.
  */
  //img[i].units                   = "1 4";
  //img[i].filename               = "ring_clump_C0_1-0.fits";   // Output filename

//INSERT IMAGES HERE

}


/******************************************************************************/

void density(double x, double y, double z, double *density){

    density[0] = grid_interpolation_3D(x, y, z, grid_density, grid_density_floor);
}


/******************************************************************************/

void temperature(double x, double y, double z, double *temperature){
  
  temperature[0] = grid_interpolation_3D(x, y, z, grid_temperature, grid_temperature_floor);

  //double r;
  //double Lstar,Tbb;

  //Lstar=8.7e0; //beta Pic

  //r=sqrt(x*x+y*y);

  //Tbb=278.3e0*sqrt(sqrt(Lstar)/(r/AU));

//printf("%f %f \n", Tbb,r);

    //temperature[1] = Tbb;

}

/******************************************************************************/


void molNumDensity(double x, double y, double z, double *nmol){

  nmol[0] = grid_interpolation_3D(x, y, z, grid_nmol, grid_nmol_floor);

}


/******************************************************************************/

void doppler(double x, double y, double z, double *doppler){
  /*
   * 200 m/s as the doppler b-parameter. This
   * can be a function of (x,y,z) as well.
   * Note that *doppler is a pointer, not an array.
   * Remember the * in front of doppler.
   */
  
  *doppler = 200.;
}


/******************************************************************************/

void velocity(double x, double y, double z, double *vel){
  //double r;
  //double G,Mstar,vkep;
  
  //G=6.67e-11;
  //Mstar=2.e30*1.75; //beta Pic

  //r=sqrt(x*x+y*y);
    
  //vkep=sqrt(G*Mstar/r);

  vel[0] = grid_interpolation_3D(x, y, z, grid_velocity_x, grid_velocity_x_floor);
  vel[1] = grid_interpolation_3D(x, y, z, grid_velocity_y, grid_velocity_y_floor);
  vel[2]=0.;
}

/******************************************************************************/
