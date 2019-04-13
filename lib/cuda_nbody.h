// ##########################################################################

double cuda_vdw_force(double xj[][3], // position of j-th particles
			 int mj[],    // mass of j-th particles
    double xi[][3], // position of i-th particles
    int mi[],    // mass of i-th particles
    double params[],    // softening parameter
    double ai[][3], // force of i-th particles
    int ni,         // number of i-th particles
    int nj);        // number of j-th particles

double cuda_vdw_pot(double xj[][3], // position of j-th particles
			 int mj[],    // mass of j-th particles
    double xi[][3], // position of i-th particles
    int mi[],    // mass of i-th particles
    double params[],    // softening parameter
    double *pot,
    int ni,         // number of i-th particles
    int nj);        // number of j-th particles

int cuda_vdw_force_setsmart(int i);
int cuda_vdw_pot_setsmart(int i);

// ##########################################################################

double cuda_coulomb1_jerk(double xj[][3],double vj[][3], // position of j-th particles
			 int mj[],    // mass of j-th particles
    double xi[][3], double vi[][3],// position of i-th particles
    int mi[],    // mass of i-th particles
    double params[],    // softening parameter
    double ai[][3], // force of i-th particles
    int ni,         // number of i-th particles
    int nj);        // number of j-th particles

double cuda_coulomb1_force(double xj[][3], // position of j-th particles
			 int mj[],    // mass of j-th particles
    double xi[][3], // position of i-th particles
    int mi[],    // mass of i-th particles
    double params[],    // softening parameter
    double ai[][3], // force of i-th particles
    int ni,         // number of i-th particles
    int nj);        // number of j-th particles

double cuda_coulomb1_pot(double xj[][3], // position of j-th particles
			 int mj[],    // mass of j-th particles
    double xi[][3], // position of i-th particles
    int mi[],    // mass of i-th particles
    double params[],    // softening parameter
    double *pot,
    int ni,         // number of i-th particles
    int nj);        // number of j-th particles

int cuda_coulomb1_jerk_setsmart(int i);
int cuda_coulomb1_force_setsmart(int i);
int cuda_coulomb1_pot_setsmart(int i);

// ##########################################################################

double cuda_ewald3d_pot_Rspace(
    double xj[][3],  // position of j-th particles
    double mj[],     // mass of j-th particles
    double xi[][3],  // position of i-th particles
    double mi[],     // mass of i-th particles
    double params[], // softening parameter
    double *pot,
    int ni,          // number of i-th particles
    int nj);         // number of j-th particles

double cuda_ewald3d_force_Rspace(
    double xj[][3],  // position of j-th particles
    double mj[],     // mass of j-th particles
    double xi[][3],  // position of i-th particles
    double mi[],     // mass of i-th particles
    double params[], // softening parameter
    double ai[][3],  // force of i-th particles
    int ni,          // number of i-th particles
    int nj);         // number of j-th particles

double cuda_ewald3d_force_Kspace(
    double xj[][3],  // position of j-th particles
    double mj[],     // mass of j-th particles
    double xi[][3],  // position of i-th particles
    double mi[],     // mass of i-th particles
    double params[], // softening parameter
    double ai[][3],  // force of i-th particles
    int ni,          // number of i-th particles
    int nj);         // number of j-th particles

double cuda_ewald3d_force_D(
    double xj[][3],  // position of j-th particles
    double mj[],     // mass of j-th particles
    double xi[][3],  // position of i-th particles
    double mi[],     // mass of i-th particles
    double params[], // softening parameter
    double ai[][3],  // force of i-th particles
    int ni,          // number of i-th particles
    int nj);         // number of j-th particles

int cuda_ewald3d_pot_Rspace_setsmart(int i);
int cuda_ewald3d_force_Rspace_setsmart(int i);
int cuda_ewald3d_force_Kspace_setsmart(int i);
int cuda_ewald3d_force_D_setsmart(int i);

// ##########################################################################

#define SECEL_TABLE_DIM 5
double cuda_secel_raw_sel(double xj[][3], double xi[][3], double yi[][3], double mi[], int a[][SECEL_TABLE_DIM], int ni, int nj);
int cuda_secel_raw_sel_setsmart(int i);

// ##########################################################################

int cucutils_deviceQuery(void);
int cucutils_devinfo( int ForceDev, char *name, int *multiproc, int *coresperblock,
		     int *globalmem, int *constmem,int *sharedmemperblock,int *numregperblock);
int cucutils_setgetdev(int dev);

//void cunbody_dumptime(void);



