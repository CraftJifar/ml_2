% This is ML assignment 1 - simulate single layer NN
% By Jifar Mekonnen , Id: GSR/3602/11

%init the weight matrix by generating random numbers between 0 and 1
% fprintf("Iteration No: "+0 +" , weight at initalization stage is: ");
W1 = rand(2,11);
W2 = rand(3,3);

%define the desired vector
desired = [3.4, 9.6, 5.1]';
desired = desired/ norm(desired);


%define the input vector
x1 = [6.4, 1.3, 2.9, 19.7, 12.5, 2.0, 6.2, 14.9, 5.0, 1.7];
x1 = x1/ norm(x1);
x1 = [1, x1]';


% define the acceptable error level , error_min
Error_min = 0.00001;

% define Beta , Beta
Beta = 1;

% define Neu,
Neu = 0.65;

% define vector to store each iteration errors
errors = zeros(1, 2000);

for i = 1 : 2000
    
    
    % For input X1
   
    % U1
    U1 = W1 * x1;
    % V 
    V = [1;sigmoid(U1)];
    % U2 
    U2 = W2 * V;
    % Y
    Y = sigmoid(U2);
    % E
    E1 = Y - desired;
    E1S = E1.^2;
    
    
    
    % layer 2 learn
    Size_W2 = size(W2);
    Temp_W2 = zeros(Size_W2(1), Size_W2(2));
    for p = 1 : Size_W2(1)
        for q =  1: Size_W2(2)
            Temp_W2(p,q) = W2(p,q) - ( Neu .* E1(p) .* sigmoidDervetive(U2(p)) * V(q)) ;        
        end
    end
    
    % layer 1 learn 
    Size_W1 = size(W1);
    Temp_W1 = zeros(Size_W1(1), Size_W1(2));
    for u = 1 : Size_W1(1)
        for v = 1 : Size_W1(2)
            Sum_s = 0;
            for s = 1 : Size_W2(1)
                Sum_s_temp = (E1(s) .* sigmoidDervetive(U2(s)) .* W2(s,u));
                Sum_s = Sum_s + Sum_s_temp;
            end
            Grad = Sum_s .* sigmoidDervetive(U1(u)) .* x1(v);
            Temp_W1(u,v) = W1(u,v) - Neu .* Grad;
        end
    end
    
   
    
    E = 0.5 * sum(E1S);
    
    
    errors(i) = E;
    
    plot(1:i,errors(1:i));
    title('Value of error on each iteration');
    xlabel('Iteration');
    ylabel('Error');

    
    if E <= Error_min
        fprintf("\nIteration No: "+ i +" , weight after succcessful training is: ")
        W1
        W2
        break;
    end
    
    
%     
    W1 = Temp_W1;
    W2 = Temp_W2;
    
end



