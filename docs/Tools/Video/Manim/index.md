## Workflow

1. render manim animation
2. import into video editor
3. record voiceover
4. use lossless cut/shutter encoder/ffmpeg to attach the audio

## Installation

make sure to add everything to system path

Advanced System Settings > Environmental Variables > Path > Add > folder

## Dependencies

1. python
   python using anaconda
2. latex
3. manim
   1. `conda install pycairo`
   2. `pip install manim`
4. ffmpeg

## Macos

``` bash
python -m pip install -r ~/Desktop/manim-master/requirements.txt
```

install latex mactex

ffmpeg

``` bash
sudo chown -R $(whoami) /usr/local/bin
mv ffmpeg ffplay ffprobe /usr/local/bin
```

## Text

Tex for text

MathTex for math

`self.play(Write(t1), run_time = 5), self.wait(delay)`

## Matrices

```python
Matrix([
  ("10", "20", "45"),
	("45", "97  ", "123"),
	("133", "56", "75")],
  size= 0.5
)
```

## Screen

`self.clear()` clears screen

## Example Program

## Test

``` python
from manim import *

## delay in seconds
delay = 0.1 ## bw animations
endDelay = 2 ## end 

class testAnimation(Scene):
    def construct(self):
        t1 = Tex("Introduction to Equations", color = BLUE).to_corner(UP)
        self.play(Write(t1), run_time = 5), self.wait(delay)
        
        t2 = MathTex("x^2 + 2x").to_corner(UP).next_to(t1, DOWN)
        self.play(Write(t2)), self.wait(delay)

        t3 = Tex("A basic quadratic equation", color = GOLD_B).to_corner(RIGHT)
        self.play(Transform(t1, t3)), self.wait(delay)

        t4 = t3.shift(DOWN)
        self.play(Transform(t1, t4)), self.wait(endDelay)

        page1 = VGroup(t1, t2)

        ## self.clear()
        
        t5 = MathTex(r"h(x) = \theta X \text{ or } X\theta")
        
        self.play(Write(t5)), self.wait(endDelay)

        page2 = VGroup(t5)
```

## Poisson Dist

``` python
from manim import *

## sizing
h1 = 2
h2 = 1.5
pwidth = 12
pline  = 1.2

## delay in seconds
delay = 0.1 ## b/w animations
endDelay = 2 ## slide end

class TestAnimation(Scene):
    def construct(self):
        ## page 1
        t = [0] ## start array from 1

        t.append(Tex("Introduction to Statistics", color = PINK).to_corner(UP)), t[-1].scale(h1)
        t.append(Tex("Poisson Distribution", color = GOLD_B).next_to(t[1], DOWN)), t[-1].scale(h2)
        t.append(Text("""
                The Poisson Distribution is a discrete probability distribution
                that measures the probability that a certain number of independent 
                events occur within a certain interval/continuum.
                """, line_spacing= pline) )
        t[-1].width = pwidth

        for i in range(1, 3+1): ## 1 - 3
            if(i == 3):
                self.play(Write(t[i]), run_time = 10)
            else:
                self.play(Write(t[i]))
            self.wait(delay)
        self.wait(endDelay)

        ## page 2
        self.clear()
        t = [0]

        t.append(Tex("Formula", color =  GOLD_B).to_corner(UP)), t[1].scale(h1)
        t.append(MathTex(r"P\left( x \right) = \frac{ {e^{ - \mu } \mu ^x } }{ {x!} }"))
        t.append(MathTex("x = 0, 1, 2, ...").next_to(t[2]))

        for i in range(1, 3+1):
            if (i == 2):
                self.play(Write(t[i]), run_time = 3), self.wait(delay)
                t[i].generate_target()
                t[i].target.shift(LEFT * 2)
                self.play(MoveToTarget(t[i])), self.wait(delay)
            else:
                self.play(Write(t[i]))
            self.wait(delay)
        self.wait(endDelay)

        ## thank you
        self.clear()

        t = Tex("Thank you", color = GOLD_B)
        t.scale(h1)
        self.play(Write(t)), self.wait(endDelay)
```

## Example

```python
from manim import *

## sizing
h1 = 2
h2 = 1.5
pwidth = 12
pline  = 1.2

## delay in seconds
delay = 0.1 ## b/w animations
endDelay = 2 ## slide end

class TestAnimation(Scene):
  def construct(self):
      ## page 1
      t = [0] ## start array from 1

      t.append(Tex("Introduction to Statistics", color = PINK).to_corner(UP)), t[-1].scale(h1)
      t.append(Tex("Poisson Distribution", color = GOLD_B).next_to(t[1], DOWN)), t[-1].scale(h2)
      t.append(Text("""
              The Poisson Distribution is a discrete probability distribution
              that measures the probability that a certain number of independent 
              events occur within a certain interval/continuum.
              """, line_spacing= pline) )
      t[-1].width = pwidth

      for i in range(1, 3+1): ## 1 - 3
          if(i == 3):
              self.play(Write(t[i]), run_time = 10)
          else:
              self.play(Write(t[i]))
          self.wait(delay)
      self.wait(endDelay)

      ## page 2
      self.clear()
      t = [0]

      t.append(Tex("Formula", color =  GOLD_B).to_corner(UP)), t[1].scale(h1)
      t.append(MathTex(r"P\left( x \right) = \frac{ {e^{ - \mu } \mu ^x } }{ {x!} }"))
      t.append(MathTex("x = 0, 1, 2, ...").next_to(t[2]))

      for i in range(1, 3+1):
          if (i == 2):
              self.play(Write(t[i]), run_time = 3), self.wait(delay)
              t[i].generate_target()
              t[i].target.shift(LEFT * 2)
              self.play(MoveToTarget(t[i])), self.wait(delay)
          else:
              self.play(Write(t[i]))
          self.wait(delay)
      self.wait(endDelay)

      ## thank you
      self.clear()

      t = Tex("Thank you", color = GOLD_B)
      t.scale(h1)
      self.play(Write(t)), self.wait(endDelay)
```

## Running

### CLI

```bash
manim -hq -p file.py Scene_Name -w test.mp4

manimgl -o -f file.py Scene_Name -w test.mp4
```

- `manim test.py -p -ql``
- ``manim test.py -p -ql`
- manim test.py -p -qh`

### Within `.py` script

```python
command = "manim -hq -p file.py Scene_Name"

import subprocess
subprocess.Popen(command)
```