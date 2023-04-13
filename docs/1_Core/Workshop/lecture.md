# Lecture


## Ch1 Manufacturing Processes:  
  
Casting : Material is given desired shape by melting.  
Machining : The process of removing unwated material from the surface of a material.  
Forming : The process which involves the deformation of a substance by going beyond its yeild strength to obtain desired shape.  
Powder Metallurgy : Fine powdered materials are pressed into desired shapes + heated and are placd in controlled environements to bond and get the finished product.  
Joining : Two or more pieced are joined together to produce the required shape. Two types are permenant joining and temporary joining.   

---

## Ch2 Engineering Materials:

Loading : Tensile Loading (pull) + Compressive Loading (Push)

Stress and Strain :  
- Stress : Sujected to external/resisting forces.  
    - Tensile
    - Compressive
    - Shear  
- Strain : Ratio of change in dimension to original dimension. 

Poisson's Ratio = $\frac{\text{Lateral Strain}}{\text{Longitudinal Stress}}$

Toughness vs Hardness vs Resilience  
| Property   | Definition                                                                                               | Measurement                                                                |
|------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Toughness  | Measure of Energy a material can absorb before it fractures. It is measured by the area under the stress-strain curve. | Area under the stress-strain curve                                          |
| Hardness   | Property of a surface to resist abrasion or indentation.                                                  | -                                                                         |
| Resilience | Capacity of a material to absorb energy elastically. Upon removal of the load the energy stored is given off. | Triangular area under the elastic portion of the stress-strain curve.    |


Creep vs Fatigue :

- Creep: Time-dependent failure due to prolonged time under load. 
- Fatigue: Unexpected and sudden failure that can occur under the yield point.

---

## Ch3 Measurments in Manufacturing:

Metrology : Science and process of ensuring the measurement meets specified degrees of both accuracy and precision. 

Measurement : Is the process of comparision of an unknown quanitity with a known quantity.

Inspection : Examination of a component to determine wether it meets specified needs or not.

Gauging : The proces of determining wheter the dimension is within specified limits or not. 

Testing : The process to know the perofmance of a product. 

Accuracy : The closeness of the measured value to the true value.
 
Precision : The closeness of two or measured values to each other. 

Tolerance : The permissible deviation of a dimension from the desired size is known as tolerance. 

Surface Finish : The amount of geometric irregularities produced on the surface of a component during a manufacturing process.  

---

## Ch4 Material Removal Process:

- Mechanism of Material Removal:  
- Depth of Cut  
- Feed - By how much distance the cutting tool must advance for each revolution of work.  
- Cutting Speed 
- Cutting Speed : The speed at which the workpiece moves with respect to the tool.  
S = Pi * D(mm) * N / 1000 
- Depth of Cut: D1-D2/2  
- Roughing Operation : A large chunk to be removed without considering perfection.  
- Finishing Operation : Only a small portion is removed keeping final touches in mind.
- MRR : 1000 * V * d * f  
- Turning : Excess material is removed by giving a depth of cut to its diameter.  
- Facing : Used to cut a flat surface perpendicular to the work peice's rotational axis. It is used to reduce the length of the workpiece. The length the workpiece travels is called the radius of the job. The depth of the cut is along the axis of the job.  
- Knurling : To produce regular patterns on the surface of metals. It is the process of pressing the metal hard enough to cause plastic deformation of metal into peaks and troughs. Low cutting speed and feed can be used w/ plenty of coolant. MRR is very low.  
- Grooving : Narrow grooves on Cylindrical Shapes, the diameter of the surface is slightly reduced. Cutting speed is slow. Depth of cut is given but no feed.  
- Parting : It is the operation of cutting a workpiece into 2 parts. The workpiece is rotated at a slow speed and the parting tool is fed perpendicular. NOTE : If a slow feed is used it wil run for 2-3 revolutions without cutting and will suddenly bite the machine, this is undesired and is called hogging.  
- Chamferigng : It is the operation of bevelling (smoothening) the sharp edges of a workpiece to avoid an injuries. It is used at an angle of 45degs.

---
```mermaid
graph LR;
    A[Mechanism of Material Removal]-->|starts with|B[Depth of Cut];
    A-->|starts with|C[Feed];
    A-->|starts with|D[Cutting Speed];
    D-->|formula|E[S = Pi * D(mm) * N / 1000];
    B-->|formula|F[D1-D2/2];
    A-->|type of|G[Roughing Operation];
    A-->|type of|H[Finishing Operation];
    A-->|formula|I[MRR : 1000 * V * d * f];
    A-->|type of|J[Turning];
    A-->|type of|K[Facing];
    K-->|purpose|L[Reduce length];
    K-->|direction|M[Depth along axis];
    A-->|type of|N[Knurling];
    N-->|effect|O[Plastic deformation of metal];
    N-->|recommendation|P[Low cutting speed];
    N-->|recommendation|Q[Low feed];
    N-->|result|R[MRR is very low];
    A-->|type of|S[Grooving];
    S-->|shape|T[Narrow grooves];
    S-->|effect|U[Diameter slightly reduced];
    S-->|recommendation|V[Slow cutting speed];
    S-->|direction|W[Given depth of cut];
    A-->|type of|X[Parting];
    X-->|purpose|Y[Workpiece into 2 parts];
    X-->|rotation speed|Z[Slow rotation];
    X-->|cutting direction|AA[Perpendicular feed];
    AA-->|undesired effect|AB[Undesired hogging];
    A-->|type of|AC[Chamfering];
    AC-->|effect|AD[Bevelling of sharp edges];
    AC-->|angle|AE[Angle of 45degs];
```