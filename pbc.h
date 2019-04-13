

//extern int  periodicbc(double *q1 , double (*cq1)[3] , double (*cpyaq)[3] , double *potential , double *tpot ,  double r0 , int N , double AP , int Multipole , int select) ;
//extern int periodicbc(double *q , double (*cq)[3] , double (*cpyaq)[3] , double *potential , double *tpot , double alpha , int N , double a , double b , double c , int rcut , int kcut , int select , int gpu , double rp) ;
extern int periodicbc( int N , double *q , double (*cq)[3] , double a , double b , double c ,double (*cpyaq)[3] , double *potential , double *tpot , double rp ,double alpha , int rcut , int kcut , int select , int gpu , int min_img ) ;


