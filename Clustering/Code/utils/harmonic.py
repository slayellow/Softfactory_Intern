import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def convert_png_jpg(filename):
    """Generate .jpg file and Remove .png file

    Parameters
    ----------
    filename: .png filename

    Returns
    -------
    """
    im = Image.open(filename)
    rgb_im = im.convert('RGB')
    rgb_im.save(filename + '.jpg')
    if os.path.isfile(filename):
        os.remove(filename)
        print(filename + ' Remove!')

ts=0.0
te=1.0
fz=8.0
A=600000
sampling=1000 # Hz
t=np.arange(ts, te, 1/128)
Desired= A*np.sin(np.pi*fz*t)
Desired_3 = 1/3*A*np.sin(np.pi*fz*3*t)
Desired_5 = 1/5*A*np.sin(np.pi*fz*5*t)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(range(128),Desired, color="r", linewidth=1, label="Original")
plt.plot(range(128),Desired_3, color="g", linewidth=1, label="Harmonic3")
plt.plot(range(128),Desired_5, color="b", linewidth=1, label="Harmonic5")
plt.plot(range(128),Desired+Desired_3+Desired_5, color='black', linewidth=1.5, label='Original + Harmonic3 + Harmonic5')
plt.legend()
plt.xlim(0, 130)
xtick = [i * 32 for i in range(1, 5)]
ytick = [i * 100000 for i in range(-8, 9)]
plt.xticks(xtick)
plt.yticks(ytick)
ax.set_xlabel('32sample/cycle')
ax.set_ylabel('Voltage(V)')
plt.savefig('Harmonic.png')

convert_png_jpg('Harmonic.png')
plt.show()