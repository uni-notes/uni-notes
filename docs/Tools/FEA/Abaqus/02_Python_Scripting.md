# Python Scripting

This assumes that you are proficient at Python

https://www.youtube.com/watch?v=lesRkx0aPeA&list=PL8JBtohft2fuXL6zj1ZETOTBtM8ydfMd6&index=4

## Why Scripting?

- Interact with underlying Abacus data structure
- Allows to extend out-of-box functionality
- Automate repetitive pre/post-processing taks
- Mathematically post-process FE data
- Custom GUI
- Perform parametric studies

## Run Python Scripts

| Mode |  |                                 | Free | Limitations | Error Handling |
| ------------ | ------------ | ------------------------------- | ---- | ------------ | ------------ |
| CLI | Python-Only  | `abaqus python script.py`       | ✅    | Not all features are supported :/ | Stop |
| CLI | CAE + Viewer | `abaqus cae noGUI=script.py`    | ❌    |  | Stop |
| CLI | Viewer-Only  | `abaqus viewer noGUI=script.py` | ❌    |  | Stop |
| CLI |  | `abaqus cae replay=abaqus.rpy` | ❌ | | Ignore & continue |
| CLI | | `abaqus cae recover=my_model.jnl` | ❌ | |  |
| GUI | Menu | `File > Run Script` | ❌   |  | Stop |
| GUI | Abacus CLI | `excefile("script.py")` | ❌    |  | Stop |

## Basics

```python
# imports
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

# setup
executeOnCaeStartup()

# Create model
model = mdb.Model(name="Model A")

# creating sketch
s = myModel.ConstrainedSketch(
	name = "__profile__",
  sheetSize = 200.0
)

# Drawing model sketch
s.Line(
	point1=(0.0, 0.0),
  point2=(0.0, 1.0)
)
s.Line(
	point1=(0.0, 1.0),
  point2=(1.0, 1.0)
)
s.Line(
	point1=(1.0, 1.0),
  point2=(1.0, 0.0)
)
s.Line(
	point1=(1.0, 0.0),
  point2=(0.0, 0.0)
)

# Creating part object
p = (
  model
  .Part(
    name = "rect_beam",
    dimensionality = THREE_D,
    type=DEFORMABLE_BODY
  )
)

# extrude sketch to get the part
p.BaseSolidExtrude(
	sketch=s,
  depth=20.0
)
s.unsetPrimaryObject()

# setting part to the viewport
(
  session
  .viewports["Viewport: 1"]
  .setValues(displayedObject=p)
)

# clear
del model.sketches["__profile__"]
```

```python
# imports
from abaqus import *
from abaqusConstants import *
import visualizatino

# opening the db
my_odb = (
	visualization
  .openOdb(path = "beam_model.odb")
)

viewport = session.viewports[session.currentViewPortName]
(
  viewport
  .setValues(displayedObject=my_odb)
)

# accessing the step-1 from the ODB
mystep = my_obd.steps["Step-1"]

# accessing the frames of step1
frame1 = mystep.frames[-1]
frame2 = mystep.frames[-2]

disp1 = frame1.fieledOutputs["U"]
disp2 = frame2.fieledOutputs["U"]

stress1 = frame1.fieledOutputs["S"]
stress2 = frame2.fieledOutputs["S"]

deltaDisp = disp2 - disp1
deltaStress = stress2 - stress1

(
  viewport
  .obdDisplay
  .setDeformedVariables(deltaDisp)
)

# Plotting thecontour for the new data
(
  viewport
  .obdDisplay
  .setPrimaryVariable(
	  field = deltaStress,
    outputPosition = INTEGRATION_POINT,
    refinement = (INVARIANT, "Mises")
  )
)

viewport.odbDispaly.display.setValues(
	plotState=(CONTOURS_ON_DEF)
)
```

```python
# saving data to a text file
with open("delta_displacement.dat", "w") as fout:
  fout.write("%8d, %15.8E, %15.8E, %15.8E\n" % tuple([value.nodeLabel, ] + list(value.data)))
```

## Step

Attributes

- Name
- Step number
- nlgoem

Methods

- getFrame
- frames
- setDefaultField

```python
(
  odb
  .steps["step-1"]
  .frames[1]
  .fieldOutputs["U"]
  .values[0]
  .data
)
```

## Abaqus Objects

### Session

- odbs
- defaultOdbDisplay
- displayGroups
- colors
- printOptions
- psOptions
- epsOptions
- pngOptions
- xyPlots
- animationController
- views
- viewports
- paths

### Mdb

- models
  - Model
    - amplitudes
    - parts
    - interactions
    - loads
    - materials
    - steps
    - sections
    - rootAssembly
- jobs

### Odb

|                                         | `import visualization`<br />`openOdb()` | `import odbAccess`<br />`openOdb()` |
| --------------------------------------- | --------------------------------------- | ----------------------------------- |
| Use in Abacus CAE                       | ✅                                       | ✅                                   |
| Use in Abacus Python                    | ❌                                       | ✅                                   |
| Multiple ODB accessible simultaneously? | ✅                                       | ❌                                   |
| Free?                                   | ❌                                       | ✅                                   |


- rootAssembly
- parts
- sectionCategories
- steps
  - name
  - nlgoem
  - frames
  - getFrame()
  - setDefaultField()
  - getHistoryRegion()
- save()
- close()

## Journalling options

```python
(
  session
  .journalOptions
  .setValues(
  	replayGeometry=COORDINATE,
    recoverGeometry=COORDINATE
  )
)
```

|                                 |                                | Human-Readable |
| ------------------------------- | ------------------------------ | -------------- |
| Compressed Index<br />(default) | `getSequenceFromMask`          | ❌              |
| Coordinate                      | `findAt()`<br />`getClosest()` | ✅              |
| Index                           | Use entity index               | ❌              |

### Compressed Index Mode (default)

```python
all_cells = (
  mdb
  .models["Model A"]
  .parts["rect_beam"]
  .cells
)
selected_cells = (
  all_cells
  .getSequencFromMask(
    mask=("[#1 ]", )
  )
)
```

## Data Types

### Symbolic Constants

```python
from abaqusConstants import *
```

### Abaqus Boolean

`ON, OFF`

### Repositories

## Export

### Images

```python
(
  session
  .printToFile(
    fileName = "export",
    format = PNG,
    canvasObjects = (
    	session.viewports["Viewport: 1"], 
    )
  )
)
```

## Field Report

```python
session.writeFieldReport(
	filname="report.rpt",
  append=ON,
  sortItem="Node Label",
  odb=odb,
  step=0,
  frame=0,
  outputPosition=NODAL,
  variable=((
    "S", INTEGRATION_POINT
  )),
)
```

