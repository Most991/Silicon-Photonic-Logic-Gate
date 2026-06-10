function XOR_results_params
clc; clear; close all;

% ============================================================
% Fixed NAND parameters from the RESULTS section
% ============================================================
p.theta1 = 5.514948;
p.theta2 = 2.373355;
p.phi1   = 4.942147;
p.phi2   = 6.019105;
p.b      = 1.0;      % practical bias value
p.T0     = 0.70;
p.fanout = 1/sqrt(2);     % physical 3 dB split of P into Q and R branches
p.C      = (1/sqrt(2)) * [1, -1i; -1i, 1];

% Gain saturation parameters (Case 2)
gs.G0   = 8.0;
gs.Isat = 0.25;

inputs = [0 0;
          0 1;
          1 0;
          1 1];

target_xor = [0; 1; 1; 0];

% ============================================================
% Reference intensity for Case 1 normalization
% ============================================================
I00 = nand_intensity(0, 0, p);
I10 = nand_intensity(1, 0, p);
I01 = nand_intensity(0, 1, p);
Iref = max([I00, I10, I01]);

% ============================================================
% CASE 1: THRESHOLD RESTORATION
% ============================================================
fprintf('\n================ CASE 1: THRESHOLD RESTORATION ================\n');
fprintf(' A B | XOR_norm   XOR_logic   target\n');
fprintf('----------------------------------------\n');

xor1_norm  = zeros(4,1);
xor1_logic = zeros(4,1);

for n = 1:4
    A = inputs(n,1);
    B = inputs(n,2);

    [~, P_logic] = nand_gate_logic(A, B, p, Iref);
    [~, Q_logic] = nand_gate_logic(A, P_logic, p, Iref);
    [~, R_logic] = nand_gate_logic(B, P_logic, p, Iref);
    [XOR_norm, XOR_logic] = nand_gate_logic(Q_logic, R_logic, p, Iref);

    xor1_norm(n)  = XOR_norm;
    xor1_logic(n) = XOR_logic;

    fprintf(' %d %d |  %.6f      %d         %d\n', ...
        A, B, XOR_norm, XOR_logic, target_xor(n));
end

% ============================================================
% CASE 2: GAIN WITH SATURATION
% ============================================================
fprintf('\n================ CASE 2: GAIN WITH SATURATION ================\n');
fprintf(' A B | XOR field                          | |XOR|^2\n');
fprintf('---------------------------------------------------------------\n');

xor2_I = zeros(4,1);

for n = 1:4
    A = inputs(n,1);
    B = inputs(n,2);

    out = xor_gain_saturation(A, B, p, gs);
    XorField = out.XOR;
    xor2_I(n) = abs(XorField)^2;

    fprintf(' %d %d | % .6f %+.6fj | %.9e\n', ...
        A, B, real(XorField), imag(XorField), xor2_I(n));
end

xor2_norm = xor2_I / max(xor2_I);
fprintf('Normalized |XOR|^2 = ');
disp(xor2_norm.');

% ============================================================
% CASE 3: NO RESTORATION
% ============================================================
fprintf('\n================ CASE 3: NO RESTORATION ======================\n');
fprintf(' A B | XOR field                          | |XOR|^2\n');
fprintf('---------------------------------------------------------------\n');

xor3_I = zeros(4,1);

for n = 1:4
    A = inputs(n,1);
    B = inputs(n,2);

    out = xor_no_restoration(A, B, p);
    XorField = out.XOR;
    xor3_I(n) = abs(XorField)^2;

    fprintf(' %d %d | % .6f %+.6fj | %.9e\n', ...
        A, B, real(XorField), imag(XorField), xor3_I(n));
end

xor3_norm = xor3_I / max(xor3_I);
fprintf('Normalized |XOR|^2 = ');
disp(xor3_norm.');

% ============================================================
% SUMMARY TABLE
% ============================================================
fprintf('\n============================= SUMMARY =============================\n');
fprintf(' A B | Case1_logic | Case1_norm | Case2_norm | Case3_norm | target\n');
fprintf('-------------------------------------------------------------------\n');
for n = 1:4
    fprintf(' %d %d |     %d      |   %.6f  |   %.6f  |   %.6f  |   %d\n', ...
        inputs(n,1), inputs(n,2), ...
        xor1_logic(n), xor1_norm(n), xor2_norm(n), xor3_norm(n), target_xor(n));
end
fprintf('-------------------------------------------------------------------\n');

% ============================================================
% PLOTS
% ============================================================
figure;
subplot(1,3,1);
stem(1:4, xor1_norm, 'filled', 'LineWidth', 1.5);
title('Case 1: Threshold restoration', 'FontSize', 14);
xlabel('Input case');
ylabel('Normalized output');
set(gca, 'XTick', 1:4, 'XTickLabel', {'00','01','10','11'});
grid on;

subplot(1,3,2);
stem(1:4, xor2_norm, 'filled', 'LineWidth', 1.5);
title('Case 2: Gain saturation', 'FontSize', 14);
xlabel('Input case');
ylabel('Normalized |XOR|^2');
set(gca, 'XTick', 1:4, 'XTickLabel', {'00','01','10','11'});
grid on;

subplot(1,3,3);
stem(1:4, xor3_norm, 'filled', 'LineWidth', 1.5);
title('Case 3: No restoration', 'FontSize', 14);
xlabel('Input case');
ylabel('Normalized |XOR|^2');
set(gca, 'XTick', 1:4, 'XTickLabel', {'00','01','10','11'});
grid on;

end

% ============================================================
% CASE 1: NAND gate with threshold restoration
% ============================================================
function [y_norm, y_logic] = nand_gate_logic(x1, x2, p, Iref)
    Iraw = nand_intensity(x1, x2, p);
    y_norm = Iraw / Iref;
    y_logic = double(y_norm > 0.5);
end

% ============================================================
% CASE 2: XOR with gain saturation
% ============================================================
function out = xor_gain_saturation(A, B, p, gs)
    P1 = gain_saturation_field(nand_field(A, B, p), gs.G0, gs.Isat);
    Q1 = gain_saturation_field(nand_field(A, p.fanout * P1, p), gs.G0, gs.Isat);
    R1 = gain_saturation_field(nand_field(B, p.fanout * P1, p), gs.G0, gs.Isat);
    X1 = gain_saturation_field(nand_field(Q1, R1, p), gs.G0, gs.Isat);

    out.P   = P1;
    out.Q   = Q1;
    out.R   = R1;
    out.XOR = X1;
end

% ============================================================
% CASE 3: XOR with no restoration
% ============================================================
function out = xor_no_restoration(A, B, p)
    P1 = nand_field(A, B, p);
    Q1 = nand_field(A, p.fanout * P1, p);
    R1 = nand_field(B, p.fanout * P1, p);
    X1 = nand_field(Q1, R1, p);

    out.P   = P1;
    out.Q   = Q1;
    out.R   = R1;
    out.XOR = X1;
end

% ============================================================
% One NAND block: raw field output
% ============================================================
function Y = nand_field(x1, x2, p)
    Xin = [x1; x2];

    Dtheta = [exp(-1i*p.theta1), 0;
              0, exp(-1i*p.theta2)];

    Dphi = [exp(-1i*p.phi1), 0;
            0, exp(-1i*p.phi2)];

    Zp = p.C * Dtheta * p.C * Xin;
    Z  = Dphi * Zp;

    z_total = Z(1) + Z(2) + p.b;
    Y = saturable_absorber_field(z_total, p.T0);
end

% ============================================================
% One NAND block: raw intensity output
% ============================================================
function Iout = nand_intensity(x1, x2, p)
    Y = nand_field(x1, x2, p);
    Iout = abs(Y)^2;
end

% ============================================================
% Shen saturable absorber in field domain
% ============================================================
function Y = saturable_absorber_field(Ein, T0)
    Iin = abs(Ein)^2;
    Tm  = solve_Tm(Iin, T0);
    Y   = sqrt(Tm) * Ein;
end

% ============================================================
% Solve Tm from:
% Iin = 0.5 * log(Tm/T0) / (1 - Tm)
% ============================================================
function Tm = solve_Tm(Iin, T0)
    if Iin <= 1e-15
        Tm = T0;
        return;
    end

    F = @(t) 0.5 .* log(t ./ T0) ./ (1 - t) - Iin;
    Tm = fzero(F, [T0, 1 - 1e-12]);
end

% ============================================================
% Gain with saturation
% Eout = sqrt(G0/(1 + |E|^2/Isat)) * E
% ============================================================
function Eout = gain_saturation_field(Ein, G0, Isat)
    gain = sqrt(G0 / (1 + (abs(Ein)^2)/Isat));
    Eout = gain * Ein;
end
