import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

N = 4
systemSize = 50

a = qt.destroy(systemSize)
aDagger = qt.create(systemSize)

gamma1 = 1
gammaN = gamma1 / 10
beta = 3
delta = 0.4 * gamma1


alpha =  0.5 * beta
alpha = alpha * (np.cos(2 * np.pi / 9) + (1j * np.sin(2 * np.pi / 9)))

phi = qt.coherent(systemSize, alpha = alpha)

identity = qt.qeye(systemSize)

def defineLiouvian(a, aDagger, delta, gamma1, gammaN, identity, N):
    hamiltonianPart = -1j * delta * (qt.sprepost(identity, (aDagger @ a)))
    hamiltonianPart += 1j * delta * qt.sprepost((aDagger @ a).trans(), identity)

    gamma1Part = gamma1 * qt.sprepost(aDagger, a)
    gamma1Part -= 0.5 * gamma1 * (qt.sprepost(identity, (aDagger @ a)) + qt.sprepost((aDagger @ a).trans(), identity))
    
    #beta = 3 
    #gammaNPart = gammaN * qt.sprepost((aDagger**N - beta**N), (a**N - beta**N))
    #gammaNPart -= 0.5 * gammaN * (qt.sprepost(identity, (((aDagger**N - beta**N) @ (a**N - beta**N))))  + qt.sprepost((a**N - beta**N) @ (aDagger**N - beta**N), identity))

    beta = 3  #to match paper
    gammaNPart = gammaN * qt.sprepost((aDagger**N - beta**N), (a**N - beta**N))
    gammaNPart -= 0.5 * gammaN * (qt.sprepost(identity, ((aDagger**N - beta**N) @ (a**N - beta**N))) 
                             + qt.sprepost(((aDagger**N - beta**N) @ (a**N - beta**N)).trans(), identity))

    L = hamiltonianPart + gamma1Part + gammaNPart

    return L

L = defineLiouvian(a, aDagger, delta, gamma1, gammaN, identity, N)

def returnSuperPhi(phi):
   rho = qt.ket2dm(phi)
   vec_rho = qt.operator_to_vector(rho)
   return vec_rho

phi = returnSuperPhi(phi)

#print("state vectorised")
#print(phi)
#print("superoperater")
#print(L)

print("starting me_solve")

options = qt.Options(nsteps = 10e5)
#print(options)

#time_list = np.linspace(0, 1, 100)
time_list = np.logspace(-2, 3, 10000)

result = qt.mesolve(L, phi, time_list, [], [], options = options)

states = result.states

print("doing expectation values")

# Compute expectation value of a
a_expectation = [np.abs(qt.expect(a, qt.vector_to_operator(state))) for state in states]

print("plotting")

# Plot modulus of expectation value of a
plt.plot(time_list, a_expectation, label='|<a>|')

plt.xscale("log")
plt.xlabel('Time')
plt.ylabel('|⟨a⟩|')
plt.title('$\gamma_1 = $ %s, $\gamma_n = $ %s, $\\beta = $ %s' % (str(gamma1), str(gammaN), str(beta)))
plt.legend()
plt.show()