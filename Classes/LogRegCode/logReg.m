function ww = logReg(XX,yy)
[NN, DD]=size(XX);
XX=[ones(NN,1) XX];

ww=zeros(DD+1,1);

for iter=1:1000
    mu = 1./(1+exp(-XX*ww));
    S=diag(mu.*(1-mu));
    ww = ww + inv(XX'*S*XX)*XX'*(yy-mu);
end