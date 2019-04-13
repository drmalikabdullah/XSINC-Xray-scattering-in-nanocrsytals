
function data = molecular_dynamics_1d(x1 , x2 , v , mass , q , dt , nsteps )

global e1  eps0


Force  = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) / ((x2 - x1 ) * (x2 - x1 ))   ;



v1 = v ; v2 = v ;
xi = x1 ;
xii = x2 ;
distance(1,1) = abs(xi - xii);
energy(1,1) = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) / ((x2 - x1 ))   ;

for xstep = 2 : nsteps
  
    xI = xi + (v1 * dt) + (-Force / (2*mass)) * (dt * dt) ;
    xII = xii + (v2 * dt) + (Force / (2*mass)) * (dt * dt);

    v1 = v1 + (-Force/mass) * dt;
    v2 = v2 + (Force/mass) * dt ; 
    
    xi = xI;
    xii = xII;
    
    Force  = (1/(4*pi*eps0)) * (q(xstep,1) * q(xstep,1) * e1 * e1 ) / ((xii - xi ) * (xii - xi ));
    
    distance(xstep,1) = abs(xi - xii);
    
    ke1(xstep,1) = 0.5 * mass * v1 * v1 ;  
    ke2(xstep,1) = 0.5 * mass * v2 * v2 ;
    
    pe(xstep,1)  = (1/(4*pi*eps0)) * (q(xstep,1) * q(xstep,1) * e1 * e1) / ((xii - xi ))   ;
    
    energy(xstep,1) = ke1(xstep,1) + ke2(xstep,1) + pe(xstep,1) ;
end

data.distance =  distance ;
data.energy = energy ;
data.ke1 = ke1 ;
data.ke2 = ke2 ;
data.pe = pe ;
