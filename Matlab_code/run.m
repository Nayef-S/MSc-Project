%original MATLAB code
rng('shuffle')

% numerical parameters
steps=1000;
Nx = 64;
Ny = 128;
dt = 1e-3;
E=[];
Ec=[];
Umodes=[];
betas=[];

% model parameters
L = 2*pi;
alp = 0.01;
nu = 1e-6;
beta = 4.5; 
p = 2; % hyperviscosity exponent
delta = 1.0;

dx = L/Nx;
xx = linspace(0,L-dx,Nx);
dy = L*delta/Ny;
yy = linspace(0,L*delta-dy,Ny);
[x, y] = meshgrid(xx,yy);

dkx = 2*pi/L;
kkx = [0:Nx/2,-Nx/2+1:-1]*dkx;
dky = 2*pi/L/delta;
kky = [0:Ny/2,-Ny/2+1:-1]*dky;
[kx, ky] = meshgrid(kkx,kky);
k2 = kx.^2 + ky.^2;
kp = k2.^p;
modk = sqrt(k2);
kmax = max(max(modk));

%U = Usave;
U = 0.2*fft(sin(2*yy/delta));
%U = fft(randn(1,Ny))*0;
%U(5:end) = 0;
%U(1:2) = 0;

%w = wsave;
%w = sin(x);
w = 0.*randn(Ny,Nx);

M1=exp((-nu*kp - alp)*dt);
M2=(1.0-exp((-nu*kp-alp)*dt))./(nu*kp+alp);
m1=exp((-nu*kky.^(2*p) - alp)*dt);
m2=(1.0-exp((-nu*kky.^(2*p) - alp)*dt))./(nu*kky.^(2*p) + alp);
if alp == 0
    if nu == 0
        M2 = 0*M1+dt;
        m2 = 0*m1+dt;
    end
    M2(1) = dt;
    m2(1) = dt;
end

C=(k2<=12^2).*(k2>=10^2);
C(:,1)=0;
D=C./k2;
D(1,1)=0;
C=2*C./mean(mean(D)).*Nx*Ny./(L^2*delta);
sigma=sqrt(C);

hovmoller=[];
v = -1i*kx./k2.*fft2(w);
v(1,1) = 0;
tic
for step=1:steps
    U_ = repmat(ifft(U),Nx,1)';
    Upp_ = repmat(real(ifft(-kky.^2.*U)), Nx, 1)';
    wrest = -U_.*real(ifft2(1i*kx.*fft2(w))) + ...
            -(beta - Upp_).*real(ifft2(v));
    w = real(ifft2(M1.*fft2(w) + M2.*fft2(wrest)));

    eta = (randn(Ny,Nx)+1i*randn(Ny,Nx)).*sigma/sqrt(2);
    w = w + sqrt(dt)*real(ifft2(eta));

    v = -1i*kx./k2.*fft2(w);
    v(1,1) = 0;

    aveVW = alp*mean(real(ifft2(v)).*w, 2)'*L;
    U = m1.*U + m2.*fft(aveVW);
    U(1) = 0;

    % truncated model
	U(6:end) = 0;
    U(1) = 0;

    % de-alias:
    U(abs(kky)>2/3*max(kky)) = 0;
    w = fft2(w);
    w(modk>2/3*kmax) = 0;
    w = real(ifft2(w));

    if mod(step,1000)==0
	    [a ,b] = max(abs(U(2:6)));
		 highestmode = b-1;
	
	    elapsed=toc;
		frac=step/steps;
	    fprintf('Step=%d (%d%%), dominating mode=%d, elapsed=%g, estimated=%g\n', step, round(frac*100), highestmode, elapsed, elapsed/frac)
	if mod(step, 1e5)==0
	   	    %save_stuff
		end
        u = 1i*ky./k2.*fft2(w);
        u(1,1) = 0;

        Ecoarse = 0.5*mean(real(ifft(U)).^2)*L*delta;
        Enow = Ecoarse + 0.5*alp*real(mean(mean(ifft2(v).^2 + ...
                                                ifft2(u).^2)))*L^2*delta;
        Ec = [Ec, Ecoarse];
        E=[E, Enow];
        figure(5)
        plot(E)
        hold on
        plot(Ec)
        hold off
       %ylim([0,1])
       %step
        figure(1)
        plot(yy, aveVW)
        xlabel('y')
        ylabel('\langle v w \rangle')
        figure(2)
        imagesc(xx,yy,w)
        colorbar
       %saveas(gcf, sprintf('om_%08d',step), 'png')
        figure(3)
        plot(yy, real(ifft(U)), '-r', yy, real(ifft(-kky.^2.*U))-beta, '-b');
        figure(4)
        psi=U./(1i*kky); psi(1)=0;
        hovmoller=[hovmoller;real(ifft(psi))];
        if step > 1000
            imagesc(hovmoller')
            colorbar
        end
        %mean(0.5*real(ifft(U)).^2)


        Umodes_ = abs(U(2:6));
        Umodes = [Umodes; Umodes_];
        figure(6)
        plot(Umodes, 'x-');
        legend('k_y=1', 'k_y=2', 'k_y=3', 'k_y=4', 'k_y=5', 'Location', ...
               'Best')
        drawnow
    end
end

max(ifft(U))

%Usave2 = Usave;
%wsave2 = wsave;
%Usave = U;
%wsave = w;
