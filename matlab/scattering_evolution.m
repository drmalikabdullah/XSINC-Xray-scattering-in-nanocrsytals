
function QI_matrix = scattering_evolution(mypath,startsnp,deltat,finalsnp)


qpoints = '/q_points' ;
tsteps = '/timesteps/' ;
ii = 1 ;
qpnts  = load([mypath qpoints ]);
%C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]} ;
%cc=hsv(12);
cc = lines(100); % 10x3 color list
fsum = 0 ;
figure;
hold on;
for ts = startsnp:deltat:finalsnp
    intensity = load([mypath tsteps num2str(ts,'%08d')]);
    fsum = fsum + intensity;
    plot(qpnts(:,1),intensity,'-.','color',cc(ii,:)); 
    ii = ii + 1;

end

%iii = ii - 1 ;
%Legend=cell(ii,1);
%for iter=1:ii
%    Legend{iter}=strcat('', num2str(iter));
%end
%legend(Legend)

hold off;

QI_matrix = [qpnts fsum(:,2)];
