clear all
close all

data = dlmread('heightWeightData.txt');
yy=data(:,1)-1;
xx=data(:,2:3);

%xx=xx-repmat(mean(xx,1),size(xx,1),1);
ww=logReg(xx, yy)


idx = yy==1;
plot (xx(idx,1),xx(idx,2),'b.');
hold on
plot (xx(~idx,1),xx(~idx,2),'r.');

x1=min(xx(:,1)):0.1:max(xx(:,1));
x2=(-ww(1)-ww(2).*x1)./ww(3);
hold on
plot(x1,x2,'.g');
hold off
