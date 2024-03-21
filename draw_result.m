save_path = 'result_data\6Jx45\X向实验\0.5倍额定负载\预压\20100125-17A\'
experiment_name = '20100125-17A'

axis off;
x0=0.1;y0=0.95;text(x0,y0,experiment_name);
% x0=8.5;y0=140;text(x0,y0,'10等分位移－抗力对应值');
% % 绘制原始曲线%
primitiveAcceleration_path = [save_path, 'primitiveAcceleration.csv'];
primitiveAceelerationCurve = csvread(primitiveAcceleration_path);
t = primitiveAceelerationCurve(:,1);
y = primitiveAceelerationCurve(:,2);
dt = t(2)-t(1);
N = length(t)
[ymin,il] = min(y); [ymax,i2]=max(y);
if abs(ymax)<abs(ymin) 
    ym = ymin;
    i=il;
else
    ym=ymax; i =i2;
end
yran = [-abs(ym*1.2) abs(ym*1.2)];
x = N*dt;
xran = [0 x yran];
axes('pos',[.12 .72 .28 .1]);
plot(t,y);title(' ');
xlabel('时间（ms）');ylabel('加速度（g）');axis(xran);
x0 = [0 x]; y0 = [0 0]; line(x0,y0);
set(gca,'xtick', 0:x/5:x);
set(gca, 'fontsize',8);


%绘制冲击加速度曲线%
sharkAcceleration_path = [save_path, 'sharkAcceleration.csv'];
sharkAccelerationCurve = csvread(sharkAcceleration_path);
t = sharkAccelerationCurve(:,1);
a = sharkAccelerationCurve(:,2);
[ymin, i1] = min(a); [ymax, i2] = max(a);
if abs(ymax)<abs(ymin)
    ym = ymin;
    i = i1;
else
    ym = ymax;
    i=i2;
end
yran = [-abs(ym*1.2) abs(ym*1.2)];
N = round(N/3); x= round(N*dt);
xran = [0 x yran];
axes('pos', [.12 .54 .28 .1]); plot(t, a); title(' ');
xlabel('时间（ms）');ylabel('加速度（g）');axis(xran);
x0 = [0 x]; y0=[0 0]; line(x0,y0);
set(gca, 'xtick', 0:x/5:x);
set(gca, 'fontsize', 8);
x0 = i*dt+0.5; y0=ym*1.05; h=text(x0,y0,' ');
set(h, 'fontsize', 10);

%绘制冲击速度曲线
sharkVelocity_path = [save_path, 'sharkVelocity.csv'];
sharkVelocityCurve = csvread(sharkVelocity_path);
t = sharkVelocityCurve(:, 1);
v = sharkVelocityCurve(:, 2);
[vm, i] = max(v);
yran=[-abs(vm)*1.2 abs(vm)*1.2];
xran=[0 x yran];
axes('pos',[.12 .36 .28 .1]);plot(t,v);title('   ');
xlabel('时间 (ms)');ylabel('速度 (m/s)');axis(xran);
x0=[0 x];y0=[0 0];line(x0,y0);
set(gca,'xtick',0:x/5:x);
set(gca,'fontsize',8);
v1m=[num2str(abs(vm)) ' m/s'] ;
x1=i*dt+0.5;y1=vm*1;h=text(x1,y1,' ');
set(h,'fontsize',10);

% 绘制冲击位移曲线
sharkDisplacement_path = [save_path, 'sharkDisplacement.csv'];
sharkDisplacementCurve = csvread(sharkDisplacement_path);
t = sharkDisplacementCurve(:, 1);
d = sharkDisplacementCurve(:, 2);
[smin,i1]=min(d);[smax,i2]=max(d);
if abs(smax)<abs(smin) 
    sm=smin;i=i1;
else
    sm=smax;i=i2;
end;
yran=[-abs(sm)*1.2 abs(sm)*1.2];
xran=[0 x yran];
axes('pos',[.12 .18 .28 .12]);plot(t,d);title('   ');
xlabel('时间 (ms)');ylabel('位移 (mm)');axis(xran);
x0=[0 x];y0=[0 0];line(x0,y0);
set(gca,'xtick',0:x/5:x);
set(gca,'fontsize',8);
s1m=[num2str(abs(sm)) ' mm'] ;
s2m=[num2str(abs(smin)) ' mm'] ;
x1=i1*dt+0.5;y1=smin*1.05;h=text(x1,y1,' ');
set(h,'fontsize',10);
Dm=sm;

orient tall;

% 冲击特性曲线
sharkDynamicCharacteristic_path = [save_path, 'sharkDynamicCharacteristic.csv'];
sharkDynamicCharacteristicCurve = csvread(sharkDynamicCharacteristic_path);
sharkCharacteristicParameter_path = [save_path, 'sharkCharacteristicParameter.csv'];
sharkCharacteristicParameter = importdata(sharkCharacteristicParameter_path);
sharkCharacteristicParameter = sharkCharacteristicParameter.data;


d = sharkDynamicCharacteristicCurve(:, 1);
f = sharkDynamicCharacteristicCurve(:, 2);

fm=max(abs(f(1:length(f)/1.2)));Dm1=abs(Dm)*1.2;
fran=[-abs(fm)*0.2 abs(fm)*1.2];
xran=[0 Dm1 fran];

DD = sharkCharacteristicParameter(:,1);
F = sharkCharacteristicParameter(:,2);
axes('pos',[.5 .55 .25 .3]);
h=plot(d,f,DD,F,'-',DD,F,'*');
set(h(2),'LineWidth',2);
xlabel('冲击位移( mm)');ylabel('冲击力( kN)');axis(xran);
set(gca,'fontsize',8);



% 环境参数
parameter_path = [save_path, 'EnvironmentalParameterAndExperimentalResult.csv'];
parameter = importdata(parameter_path);
parameter = parameter.data;
At = parameter(1);
H = parameter(2);
M = parameter(3);
Am  = parameter(4);
Vo = parameter(5);
Dm = parameter(6);
Fm = parameter(7);
Wm = parameter(8);
Wa = parameter(9);
Kp = parameter(10);
Ke = parameter(11);
C = parameter(12);
SF = parameter(13);
n = parameter(14);
grid;
At=['环境温度: ' num2str(At) ' ℃']; 
H=['落锤高度: ' num2str(H) ' cm '];M=['锤头质量: ' num2str(M) ' kg '];
Am=['Am=' num2str(abs(Am)) ' g '];Vo=['Vo=' num2str(Vo) ' m/s '];
Dm=['Dm=' num2str(Dm) ' mm '];Fm=['Fm=' num2str(Fm) ' kN '];
Wm=['Wm=' num2str(Wm) ' J '];Wa=['Wa=' num2str(Wa) ' J '];
Kp=['Kp=' num2str(Kp) ' kN/m '];Ke=['Ke=' num2str(Ke) ' kN/m ']; 
C1=['C=' int2str(C) ' N.s/m '];SF=['SF=' num2str(SF) '        '];
n=['η=' num2str(n)];

x1=1.2*Dm1;f0=fm*1.3*0.07;d4=0.25*Dm;
y1=fm*1.1;h=text(x1,y1,At);set(h,'fontsize',9);
y1=y1-f0;h=text(x1,y1,H);set(h,'fontsize',9);
y1=y1-f0;h=text(x1,y1,M);set(h,'fontsize',9);
x1=1.2*Dm1;y1=(y1-f0)*0.95;h=text(x1,y1,Am);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Vo);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Dm);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Fm);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Wm);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Wa);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Kp);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,Ke);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,C1);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,SF);set(h,'fontsize',8);
x1=1.2*Dm1;y1=y1-f0;h=text(x1,y1,n);set(h,'fontsize',8);
yran=[0 125];xran=[0 30 yran];
axes('pos',[.5 .15 .4 .25]);axis(xran);

x1=[0 0];y1=[0 125];x2=[0 30];y2=[0 0];
x3=[0 30];y3=[125 125];x4=[30 30];y4=[0 125];x5=[6 6];y5=[0 125];x6=[24 24];y6=[0 125];
x7=[0 30];y7=[110 110];x8=[0 30];y8=[100 100];x9=[0 30];y9=[90 90];x10=[0 30];y10=[80 80];
x11=[0 30];y11=[70 70];x12=[0 30];y12=[60 60];x13=[0 30];y13=[50 50];x14=[0 30];y14=[40 40];
x15=[0 30];y15=[30 30];x16=[0 30];y16=[20 20];x17=[0 30];y17=[10 10];x18=[12 12];y18=[0 125];x19=[19 19];y19=[0 125];
line(x1,y1);line(x2,y2);line(x3,y3);line(x4,y4);line(x5,y5);line(x6,y6);line(x7,y7);
line(x8,y8);line(x9,y9);line(x10,y10);line(x11,y11);line(x12,y12);line(x13,y13)
line(x14,y14);line(x15,y15);line(x16,y16);line(x17,y17);line(x18,y18);line(x19,y19);
set(gca,'ytick',0:10:100);
set(gca,'xtick',0:10:30);

axis off;
x0=8.5;y0=140;text(x0,y0,'10等分位移－抗力对应值');
x0=1;y0=117;h=text(x0,y0,'位移(mm)');set(h,'fontsize',9);
x0=7;y0=117;h=text(x0,y0,'抗力(kN)');set(h,'fontsize',9);
x0=13;y0=117;h=text(x0,y0,'分段刚度(kN/m)');set(h,'fontsize',9);
x0=19;y0=117;h=text(x0,y0,'上宽(kN)');set(h,'fontsize',9);
x0=25;y0=117;h=text(x0,y0,'下宽(kN)');set(h,'fontsize',9);


% 冲击特性参数
DD = sharkCharacteristicParameter(:,1);
F = sharkCharacteristicParameter(:,2);
KK =  sharkCharacteristicParameter(:,3);
FF = sharkCharacteristicParameter(:,4);
FFF = sharkCharacteristicParameter(:,5);
x0=1.5;y0=105.5;h=text(x0,y0,num2str(DD(1)));set(h,'fontsize',9);
x0=1.5;y0=95;h=text(x0,y0,num2str(DD(2)));set(h,'fontsize',9);
x0=1.5;y0=85;h=text(x0,y0,num2str(DD(3)));set(h,'fontsize',9);
x0=1.5;y0=75;h=text(x0,y0,num2str(DD(4)));set(h,'fontsize',9);
x0=1.5;y0=65;h=text(x0,y0,num2str(DD(5)));set(h,'fontsize',9);
x0=1.5;y0=55;h=text(x0,y0,num2str(DD(6)));set(h,'fontsize',9);
x0=1.5;y0=45;h=text(x0,y0,num2str(DD(7)));set(h,'fontsize',9);
x0=1.5;y0=35;h=text(x0,y0,num2str(DD(8)));set(h,'fontsize',9);
x0=1.5;y0=25;h=text(x0,y0,num2str(DD(9)));set(h,'fontsize',9);
x0=1.5;y0=15;h=text(x0,y0,num2str(DD(10)));set(h,'fontsize',9);
x0=1.5;y0=5;h=text(x0,y0,num2str(DD(11)));set(h,'fontsize',9);

x0=7.5;y0=105.5;h=text(x0,y0,num2str(F(1)));set(h,'fontsize',9);
x0=7.5;y0=95;h=text(x0,y0,num2str(F(2)));set(h,'fontsize',9);
x0=7.5;y0=85;h=text(x0,y0,num2str(F(3)));set(h,'fontsize',9);
x0=7.5;y0=75;h=text(x0,y0,num2str(F(4)));set(h,'fontsize',9);
x0=7.5;y0=65;h=text(x0,y0,num2str(F(5)));set(h,'fontsize',9);
x0=7.5;y0=55;h=text(x0,y0,num2str(F(6)));set(h,'fontsize',9);
x0=7.5;y0=45;h=text(x0,y0,num2str(F(7)));set(h,'fontsize',9);
x0=7.5;y0=35;h=text(x0,y0,num2str(F(8)));set(h,'fontsize',9);
x0=7.5;y0=25;h=text(x0,y0,num2str(F(9)));set(h,'fontsize',9);
x0=7.5;y0=15;h=text(x0,y0,num2str(F(10)));set(h,'fontsize',9);
x0=7.5;y0=5;h=text(x0,y0,num2str(F(11)));set(h,'fontsize',9);

x0=13.5;y0=105.5;h=text(x0,y0,'-');set(h,'fontsize',9);
x0=13.5;y0=95;h=text(x0,y0,num2str(KK(2)));set(h,'fontsize',9);
x0=13.5;y0=85;h=text(x0,y0,num2str(KK(3)));set(h,'fontsize',9);
x0=13.5;y0=75;h=text(x0,y0,num2str(KK(4)));set(h,'fontsize',9);
x0=13.5;y0=65;h=text(x0,y0,num2str(KK(5)));set(h,'fontsize',9);
x0=13.5;y0=55;h=text(x0,y0,num2str(KK(6)));set(h,'fontsize',9);
x0=13.5;y0=45;h=text(x0,y0,num2str(KK(7)));set(h,'fontsize',9);
x0=13.5;y0=35;h=text(x0,y0,num2str(KK(8)));set(h,'fontsize',9);
x0=13.5;y0=25;h=text(x0,y0,num2str(KK(9)));set(h,'fontsize',9);
x0=13.5;y0=15;h=text(x0,y0,num2str(KK(10)));set(h,'fontsize',9);
x0=13.5;y0=5;h=text(x0,y0,num2str(KK(11)));set(h,'fontsize',9);

x0=19.5;y0=105.5;h=text(x0,y0,num2str(FF(1)));set(h,'fontsize',9);
x0=19.5;y0=95;h=text(x0,y0,num2str(FF(2)));set(h,'fontsize',9);
x0=19.5;y0=85;h=text(x0,y0,num2str(FF(3)));set(h,'fontsize',9);
x0=19.5;y0=75;h=text(x0,y0,num2str(FF(4)));set(h,'fontsize',9);
x0=19.5;y0=65;h=text(x0,y0,num2str(FF(5)));set(h,'fontsize',9);
x0=19.5;y0=55;h=text(x0,y0,num2str(FF(6)));set(h,'fontsize',9);
x0=19.5;y0=45;h=text(x0,y0,num2str(FF(7)));set(h,'fontsize',9);
x0=19.5;y0=35;h=text(x0,y0,num2str(FF(8)));set(h,'fontsize',9);
x0=19.5;y0=25;h=text(x0,y0,num2str(FF(9)));set(h,'fontsize',9);
x0=19.5;y0=15;h=text(x0,y0,num2str(FF(10)));set(h,'fontsize',9);
x0=19.5;y0=5;h=text(x0,y0,num2str(FF(11)));set(h,'fontsize',9);

x0=25.5;y0=105.5;h=text(x0,y0,num2str(FFF(1)));set(h,'fontsize',9);
x0=25.5;y0=95;h=text(x0,y0,num2str(FFF(2)));set(h,'fontsize',9);
x0=25.5;y0=85;h=text(x0,y0,num2str(FFF(3)));set(h,'fontsize',9);
x0=25.5;y0=75;h=text(x0,y0,num2str(FFF(4)));set(h,'fontsize',9);
x0=25.5;y0=65;h=text(x0,y0,num2str(FFF(5)));set(h,'fontsize',9);
x0=25.5;y0=55;h=text(x0,y0,num2str(FFF(6)));set(h,'fontsize',9);
x0=25.5;y0=45;h=text(x0,y0,num2str(FFF(7)));set(h,'fontsize',9);
x0=25.5;y0=35;h=text(x0,y0,num2str(FFF(8)));set(h,'fontsize',9);
x0=25.5;y0=25;h=text(x0,y0,num2str(FFF(9)));set(h,'fontsize',9);
x0=25.5;y0=15;h=text(x0,y0,num2str(FFF(10)));set(h,'fontsize',9);
x0=25.5;y0=5;h=text(x0,y0,num2str(FFF(11)));set(h,'fontsize',9);








