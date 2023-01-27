For simplicy, I’ve excluded

- $dx$ for pre-integration
- $+ c$ for post-integration

|                  |        Pre-Integration         |                       Post-Integration                       |                                   |
| :--------------: | :----------------------------: | :----------------------------------------------------------: | :-------------------------------: |
|    **Basic**     |        $x^n, n \ne -1$         |                    $\frac{x^{n+1}}{n+1}$                     |                                   |
|                  |         $\frac{1}{x}$          |                           $\log x$                           |                                   |
|                  |             $e^x$              |                            $e^x$                             |                                   |
|                  |             $a^x$              |                     $\frac{a^x}{\log a}$                     |                                   |
| **Coefficient**  |           $f(ax+b)$            |                    $\frac{F(ax + b)}{a}$                     |                                   |
| **Trignometric** |            $\sin x$            |                          $- \cos x$                          |                                   |
|                  |            $\cos x$            |                           $\sin x$                           |                                   |
|                  |            $\tan x$            |                       $\log \|\sec x|$                        |          $-\log\|\cos x|$          |
|                  |            $\cot x$            |                       $\log \|\sin x|$                        |     $-\log\|\text{cosec } x|$      |
|                  |            $\sec x$            |                   $\log\|\sec x + \tan x|$                    |     $-\log\|\sec x - \tan x|$      |
|                  |        $\text{cosec }x$        |               $\log\|\text{cosec } x - \cot x|$               | $-\log\|\text{cosec } x + \cot x|$ |
|                  |        $\sec x \tan x$         |                           $\sec x$                           |                                   |
|                  |    $\text{cosec }x \cot x$     |                      $-\text{cosec } x$                      |                                   |
|                  |           $\sec^2 x$           |                           $\tan x$                           |                                   |
|                  |       $\text{cosec}^2 x$       |                          $- \cot x$                          |                                   |
|     **IDK**      |    $\frac{1}{\sqrt{1-x^2}}$    |                        $\sin^{-1} x$                         |          $-\cos^{-1} x$           |
|                  |    $\frac{1}{\sqrt{1+x^2}}$    |                        $\tan^{-1} x$                         |          $-\cot^{-1} x$           |
|                  |  $\frac{1}{x \sqrt{x^2 - 1}}$  |                        $\sec^{-1} x$                         |     $- \text{ cosec}^{-1} x$      |
|   **Squares**    |     $\frac{1}{a^2 + x^2}$      |      $\frac{1}{a} \tan^{-1} \left( \frac{x}{a} \right)$      |                                   |
|                  |     $\frac{1}{x^2 - a^2}$      |       $\frac{1}{2a} \log\left\|\frac{x-a}{x+a}\right\|$        |                                   |
|                  |     $\frac{1}{a^2 - x^2}$      |       $\frac{1}{2a} \log\left\|\frac{a+x}{a-x}\right\|$        |                                   |
|  **Den Roots**   |  $\frac{1}{\sqrt{a^2 - x^2}}$  |            $\sin^{-1} \left( \frac{x}{a} \right)$            |                                   |
|                  |  $\frac{1}{\sqrt{x^2 + a^2}}$  |          $\log\left\| x + \sqrt{x^2 + a^2} \right\|$           |                                   |
|                  | $\frac{1}{x \sqrt{x^2 - a^2}}$ |       $\frac{1}{a} \sec^{-1} \left(\frac{x}{a}\right)$       |                                   |
|  **Num Roots**   |       $\sqrt{a^2 - x^2}$       | $\frac{x}{2} \sqrt{a^2 - x^2} + \frac{a^2}{2} \sin^{-1}\left(\frac{x}{a}\right)$ |                                   |
|                  |       $\sqrt{a^2 + x^2}$       | $\frac{x}{2} \sqrt{a^2 + x^2} + \frac{a^2}{2} \log \|x + \sqrt{a^2 + x^2} \|$ |                                   |
|                  |       $\sqrt{x^2 - a^2}$       | $\frac{x}{2} \sqrt{x^2 - a^2} - \frac{a^2}{2} \log \|x + \sqrt{x^2 - a^2}\|$ |                                   |
|     **IDK**      | $e^x \Big(f(x) + f'(x) \Big)$  |                          $e^x f(x)$                          |                                   |
|                  |  $x \Big(f(x) + f'(x) \Big)$   |                           $x f(x)$                           |                                   |
| **Parts/ILATE**  |         $\int (uv) dx$         |    $u \int vdx + \int \left(u' {\small \int} vdx \right)$    |                                   |

# Partial Fractions

|                   Function                   |                     Partial Fraction                      |
| :------------------------------------------: | :-------------------------------------------------------: |
|          $\frac{px+q}{(x-a)(x-b)}$           |            $\frac{A}{(x-a)} + \frac{B}{(x-b)}$            |
|   $\frac{px^2 + qx + r}{(x-a)(x-b)(x-c)}$    |   $\frac{A}{(x-a)} + \frac{B}{(x-b)} + \frac{C}{(x-c)}$   |
|            $\frac{px+q}{(x-a)^3}$            | $\frac{A}{(x-a)} + \frac{B}{(x-a)^2} + \frac{C}{(x-a)^3}$ |
|    $\frac{px^2 + qx + r}{(x-a)^2 (x-b)}$     |  $\frac{A}{(x-a)} + \frac{B}{(x-a)^2} + \frac{C}{(x-b)}$  |
| $\frac{px^2 + qx + r}{(x-a) (x^2 + bx + c)}$ |     $\frac{A}{(x-a)} + \frac{Bx + C}{(x^2 + bx + c)}$     |

# Properties

|                                      |                                     |
| :----------------------------------: | :---------------------------------: |
| $\left(\int f(x) \cdot dx \right)'$  |               $f(x)$                |
|           $\int f'(x) dx$            |             $f(x) + c$              |
|        $\int k \cdot f(x) dx$        |          $k \int f(x) dx$           |
| $\int \Big(f(x) \pm g(x) \Big) \ dx$ | $\int f(x) \ dx \pm \int g(x) \ dx$ |
