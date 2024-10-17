from cmath import pi
import matplotlib.pyplot as plt
import numpy as np

D = 1

x = np.linspace(0.55, 8, num=9)
y = np.linspace(-1.25, 1.25, num=5)
z = np.linspace(-4, 4, num=7)

# theta_probe = np.linspace(0,2*np.pi,num=36)
# d_probe = D*np.array([1.1, 1.35])

i = 0
probe = np.zeros((x.size*y.size*z.size, 3))
print('Number of probes:' + str(probe.shape[0]))
for x_probe in x:
    for y_probe in y:
        for z_probe in z:
            if (x_probe**2 + y_probe**2)**0.5 > 0.5:
                probe[i, :] = np.multiply([x_probe, y_probe, z_probe],D)
                i = i + 1

# for theta_p in theta_probe:
#     for d_p in d_probe:
#         probe[i, :] = [d_p/2*np.cos(theta_p), d_p/2*np.sin(theta_p), 0]
#         i = i + 1

# Remove zero rows
probe = probe[~np.all(probe==0,axis=1)]

f = open("system/probes", "w")
for i in range(len(probe)):
    f.write('( {x:.8f} {y:.8f} {z:.8f} )\n'.format(x=probe[i,0],y=probe[i,1],z=probe[i,2]))
f.close()


theta = np.linspace(0,2*np.pi,num=100)
xCyl = D/2*np.cos(theta)
yCyl = D/2*np.sin(theta)

fig, axs = plt.subplots(1)
plt.plot(xCyl, yCyl)
axs.set_aspect('equal', 'box')

plt.plot(probe[:,0],probe[:,1],linestyle='None',marker='o')

plt.show()

xa = 1
