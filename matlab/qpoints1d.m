

function all_qp = qpoints1d( qx , qy , qz , max_q , delta_q, filename)

qvect = [] ;
cc = 0;
cc = cc + 1 ;

delta_ql = qy - max_q;  
delta_qr = qy + max_q;


for ri = qx
    for rj = delta_ql : delta_q : delta_qr
        for rk = qz
            qvect(cc,1) = ri ;
            qvect(cc,2) = rj ;
            qvect(cc,3) = rk ;
            cc = cc + 1 ;
            end
            end
         end

filename = ['I3C-qpoints_' num2str(qx) '_' num2str(qy) '_' num2str(qz) '.txt']
         fid = fopen([filename],'w');
         fprintf(fid,'% d % d % d\n',transpose(qvect));
fclose('all');
all_qp = qvect ;
