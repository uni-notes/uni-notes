```python
X=[x1,x2,x3 ... xn]
Y=[y1,y2,y3 ... yn]
Z=[z1,z2,z3 ... zn]
for I in range (len(X)):
for J in range (len(Y)):
for K in range (len(Z)):
modelname='P1_'+ str(int(X[I]))+'_'+'P2_'+str(int(Y[J]))+'_'+'P3_'+str(int(Z[K]))
mymodel=mdb.Model(name=modelname)
```

```python
# So now you have several Models in Abaqus. Then you write or use Abaqus macro to create your script. At last, you must create a job for each model.
jobname=str(int(X[I]))+'_'+str(int(Y[J]))+'_'+str(int(Z[K]))
```

```python
mdb.Job(
  name=jobname, model=mymodel,description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,queue=None, memory=90, memoryUnits=PERCENTAGE,explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',scratch='', resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN,numDomains=1, activateLoadBalancing=False, multiprocessingMode=DEFAULT,numCpus=1
       )

mbd.JobFromInputFile(
	
)
```

```python
mdb.jobs[jobname].submit(consistencyChecking=OFF)
mdb.jobs[jobname].waitForCompletion()
```



```bash
# At last, if you want to run this script without opening Abaqus, open cmd and set a new address to the script.
cd path
```

```bash
# use this code to run Abaqus
Abaqus cae noGUI=SCRIPTNAME.py
```

