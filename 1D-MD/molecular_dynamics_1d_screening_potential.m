
function data = molecular_dynamics_1d(x1 , x2 , v , mass , q , lambda , dt , nsteps )
% molecular_dynamics_1d(x1 , x2 , v , mass , q , lambda , dt , nsteps )
% x1 = coordinate of the first particle (only x-axis)
% x2 = coordinate of the second particle (only x-axis)
% v = initial velocity
% mass = mass of the particle
% q = Array of the charges
% lambda = debye length
% dt = time step
% nsteps = total number of steps

global e1  eps0


Force  = -(1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) * (- exp(-abs(x2 - x1) / lambda(1,1)) ) * ((1 / lambda(1,1) * abs(x2 -x1)) +  (1 / (x2 - x1)^2 ) );
%Force  = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) / ((x2 - x1 ) * (x2 - x1 ))   ; Coloumb Force



v1 = v ; v2 = v ;
xi = x1 ;
xii = x2 ;
distance(1,1) = abs(xi - xii);

%energy(1,1) = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) *(1 / abs(x2 - x1 )) * exp(-abs(x2 - x1) / lambda(1,1))  ; 

%energy(1,1) = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) / ((x2 - x1 ))   ;  Coloumb Potential

energy(1,1) = (1/(4*pi*eps0)) * (q(1,1) * q(1,1) * e1 * e1) / (abs(x2 - x1 ))  * exp(-abs(x2 - x1) / lambda(1))   ;  

for xstep = 2 : nsteps
  
    xI = xi + (v1 * dt) + (-Force / (2*mass)) * (dt * dt) ;
    xII = xii + (v2 * dt) + (Force / (2*mass)) * (dt * dt);

    v1 = v1 + (-Force/mass) * dt;
    v2 = v2 + (Force/mass) * dt ; 
    
    xi = xI;
    xii = xII;
    
Force  = -(1/(4*pi*eps0)) * (q(xstep) * q(xstep) * e1 * e1) * (- exp(-abs(x2 - x1) / lambda(1,1)) ) * ((1 / lambda(1,1) * abs(xii -xi)) +  (1 / (xii - xi)^2 ) );
 %   Force  = (1/(4*pi*eps0)) * (q(xstep) * q(xstep) * e1 * e1 ) / ((xii - xi ) * (xii - xi )); Coloumb Force
   distance(xstep,1) = abs(xi - xii);
    
    ke1(xstep,1) = 0.5 * mass * v1 * v1 ;  
    ke2(xstep,1) = 0.5 * mass * v2 * v2 ;
    
   
%    pe(xstep,1)  = (1/(4*pi*eps0)) * (q(xstep) * q(xstep) * e1 * e1 ) *(1 / abs(xii - xi )) * exp(-abs(xii - xi) / lambda(xstep))  ; 
%    pe(xstep,1)  = (1/(4*pi*eps0)) * (q(xstep) * q(xstep) * e1 * e1) / ((xii - xi ))   ;   Coloumb Potential
    pe(xstep,1)  = (1/(4*pi*eps0)) * (q(xstep) * q(xstep) * e1 * e1) / ((xii - xi )) * exp(-abs(xii - xi) / lambda(xstep))  ;  
    energy(xstep,1) = ke1(xstep,1) + ke2(xstep,1) + pe(xstep,1) ;
end

data.distance =  distance ;
data.energy = energy ;
data.ke1 = ke1 ;
data.ke2 = ke2 ;
data.pe = pe ;
