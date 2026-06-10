clc; clear; close all;

T0 = 0.7;                 % initial transmittance
Iin = (0:0.001:20);          % input intensity vector
x = Iin;
Tm   = zeros(size(x));
Iout = zeros(size(x));


for k = 1:length(x)
    F = @(t) 0.5 .* log(t ./ T0) ./ (1 - t) - x(k);
    Tm(k) = fzero(F, [T0, 1 - 1e-12]);
    Iout(k) = x(k) * Tm(k);
end

Ein_pos  = sqrt(Iin);
Eout_pos = sqrt(Iout);

Ein  = [-fliplr(Ein_pos(2:end)),  Ein_pos];
Eout = [-fliplr(Eout_pos(2:end)), Eout_pos];

figure;
plot(x, Iout, 'LineWidth', 2);
xlabel('\sigma \tau_s I_{in}');
ylabel('\sigma \tau_s I_{out}');
title('Nonlinear activation function of saturable absorber', 'FontSize', 16);
grid on;

figure;
plot(Ein, Eout, 'LineWidth', 2);
xlabel('\sigma \tau_s E_{in}');
ylabel('\sigma \tau_s E_{out}');
title('Field-domain nonlinear activation function', 'FontSize', 16);
grid on;
