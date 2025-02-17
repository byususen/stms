# stms
Spatiotemporal and Multistep Smooting for Sentinel 2 Data Reconstruction

## Example Python Code

Below is an example of using stms function :

```python
#Generate data
def sine_func(x, A, B, C,D):
    y = A * np.sin(2*(np.pi/B)*(x-C)) + D
    return y
    
A = 0.3
B = 100
n_sample = 50
n_days = 400
x = np.arange(5,n_days,5)
vi_data = np.empty(0, dtype=float)
long_data = np.empty(0, dtype=float)
lati_data = np.empty(0, dtype=float)
cloud_data = np.empty(0, dtype=float)
days_data = np.empty(0, dtype=int)
id_sample = np.empty(0, dtype=object)
for i in range(n_sample):
    long = np.random.uniform(100,102,1)
    lati = np.random.uniform(-3,-1,1)
    idsamp = 'sample_' + str(i)
    C = 100 + np.random.uniform(-50,50,1)
    D = 0.5 + np.random.uniform(-0.05,0.05,1)
    for j in x:
        y = sine_func(j,A,B,C,D)
        cloud = 0.9
        y += np.random.uniform(-0.1,-0.01,1)
        cloud += np.random.uniform(-0.1,-0.01,1)
        cloud_data = np.append(vi_data,cloud)
        vi_data = np.append(vi_data,y)
        long_data = np.append(long_data,long)
        lati_data = np.append(lati_data,lati)
        days_data = np.append(days_data,int(j))
        id_sample = np.append(id_sample, idsamp)

#Add 5 random thick cloudy days for all sample
n_cloud = 5
for i in np.unique(id_sample):
    loc = np.random.choice(days_data, n_cloud,replace=False)
    for j in loc:
        vi_data[np.where(np.logical_and(id_sample == i, days_data == j))[0]] = np.random.uniform(0.1,0.2,1)
        cloud_data[np.where(np.logical_and(id_sample == i, days_data == j))[0]] = np.random.uniform(0.0,0.1,1)

#Add consecutive thick cloudy 5-10 days for all sample
for i in np.unique(id_sample):
    length_cons = np.random.randint(5,10, size=1)
    loc_first = np.random.choice(np.arange(0,len(id_sample[id_sample == i])-length_cons,1), 1,replace=False)
    loc_days = x[int(loc_first):int(loc_first + length_cons)]
    for j in loc_days:
        vi_data[np.where(np.logical_and(id_sample == i, days_data == j))[0]] = np.random.uniform(0.1,0.2,1)
        cloud_data[np.where(np.logical_and(id_sample == i, days_data == j))[0]] = np.random.uniform(0.0,0.1,1)

vi_data = stms(id_sample, days_data,vi_data,long_data,lati_data,cloud_data)
