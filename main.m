clc
clear all
close all

load('data.mat');
tracks = defaultfeatures1059tracks;
tracks(:,70) = [];
tracks(:,69) = [];
countries = xlsread('countries.xlsx');

%X = linspace(0,1059,1059);

%plot(x,defaultfeatures1059tracks(:,69),'RED');
%hold on;
%plot(x,defaultfeatures1059tracks(:,70));
%Y = sortrows(defaultfeatures1059tracks,[69 70]);
Y =[defaultfeatures1059tracks(:,69) defaultfeatures1059tracks(:,70)];
localizations = Y(:, 1) + (Y(:, 2))*i;

% Bootstraping Partición de datos 70% -> train, 30% -> test
cv = cvpartition(localizations, 'holdout', 0.3);

% Extracción de caracteristicas PCA
addpath(genpath([pwd '\sprtool']))
data.X = tracks';
data.y = countries';

%Se normalizan las características (media = 0 y desviación = 1)
X = zscore(data.X');

%Cuando se realiza reducción de dimensionalidad con PCA, la matriz 
%de características debe tener cada muestra (grabación) en las columnas 
%y cada característica en las filas
X = X';

%PCA. Se busca reducir la dimensión de los datos a 2. No se consideran 
%etiquetas de clase
model = pca(X,2);

%Se extrae las componentes principales de los datos para obtener una nueva 
%matriz de medidas con dos características
ext_data = linproj(X,model);

%Graficar nuevo espacio de característcas
% scatter3(ext_data(1,data.y==21),ext_data(2,data.y==21),ext_data(3,data.y==21),'o')
% hold on
% scatter3(ext_data(1,data.y==7),ext_data(2,data.y==7),ext_data(3,data.y==7),'ro')
% title('PCA para base de datos de emociones')
% ylabel('PCA 2')
% xlabel('PCA 1')
% legend('Emociones de baja excitación','Emociones de alta excitación') 


train = ext_data(:,training(cv));
label_train = data.y(training(cv));
%Seleccionar conjunto de prueba
test = ext_data(:,test(cv));
label_test = data.y(~training(cv));


%KNN
%Entrenar modelo con k=3
k = 5;
%Los labels deben quedar en codificación  1-of-N
label_train2 = [label_train' label_train'];
label_train2(:,1) = zeros(size(label_train2,1),1);
label_train2(:,1) = label_train2(:,2)==0;
%Entrenamiento
net = knn(size(train',2),2,k,train',label_train2);
%Prueba
[y_predict,l] = knnfwd(net,test');

subplot(2,1,1)
plot(test(1,l==1),test(2,l==1),'bo','MarkerFaceColor','b')
hold on
plot(test(1,l==2),test(2,l==2),'ro','MarkerFaceColor','r')
title('Datos de prueba clasificados por el K-NN')
xlabel('Medida 1')
ylabel('Medida 2')
legend('Emociones de baja excitación','Emociones de alta excitación')

subplot(2,1,2)
plot(test(1,label_test'==0),test(2,label_test'==0),'bo','MarkerFaceColor','b')
hold on
plot(test(1,label_test'==1),test(2,label_test'==1),'ro','MarkerFaceColor','r')
title('Datos de prueba con etiquetas reales')
xlabel('Medida 1')
ylabel('Medida 2')
legend('Emociones de baja excitación','Emociones de alta excitación')

cp = classperf(label_test', l);

% 
% 
% X = zscore(X);
% Y = zscore(Y);
% %plot(X,Y(:,69),'RED');
% %hold on;
% %plot(X,Y(:,70));
% %hold off;
% X = defaultfeatures1059tracks(:,1:68);
% Y = defaultfeatures1059tracks(:,69:70);
% 
% x = X;
% y = Y;
% y = y/norm(y);
% Y1 = Y(:,1);
% Y2 = Y(:,2);
% figure('Color','w');worldmap([min(Y1) max(Y1)],[min(Y2) max(Y2)]);geoshow('landareas.shp');geoshow(Y1,Y2,'DisplayType','point');shg;
% scatter(Y(:,2),Y(:,1));
% 
% dlmwrite('data.dat',[Y1 Y2]);
% %findcluster('data.dat');
% 
% d = 2; 
% k = 5; 
% n = 1059;
% X = Y';
% [X,label] = kmeansRnd(d,k,n); 
% y = kmeans(X,k); 
% plotClass(X,label); 
% figure; 
% plotClass(X,y); 
