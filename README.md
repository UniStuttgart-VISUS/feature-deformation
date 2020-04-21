# Feature-Based Deformation for Flow Visualization

TODO.

## Parameters

*General:*

| Parameter                 | Description                                                                           | Default value |
|---------------------------|---------------------------------------------------------------------------------------|---------------|
| Line ID                   | ID of the line to deform.                                                             | 0             |
| Vectors                   | Array name for input vectors, which will be adjusted according to the deformation.    |               |

*Smoothing:*

| Parameter                 | Description                                                                                       | Default value         |
|---------------------------|---------------------------------------------------------------------------------------------------|-----------------------|
| Method                    | Method used for straightening the input feature: direct displacement, smoothing or time-local.    | Direct displacement   |
| Variant                   | Variant of the smoothing method, either fixing the endpoints or the arc length.                   | Fix endpoints         |
| Number of iterations      | Number of smoothing steps.                                                                        | 1000                  |
| Smoothing factor          | Smoothing factor for Gaussian (or Taubin) smoothing.                                              | 0.7                   |
| Inflation factor          | Negative smoothing factor for the second step of Taubin smoothing, resulting in inflation.        | -0.75                 |

There are three different methods implemented:
- *Direct displacement:* perform a single deformation step to reach the fully straightened line,
- *Smoothing:* can be used to animate the evolution of the deformation,
- *Time-local:* restrict the deformation to the form (dx, dy, 0), assuming time on the z dimension.

*Animation:*

| Parameter                 | Description                                                                           | Default value |
|---------------------------|---------------------------------------------------------------------------------------|---------------|
| Interpolator              | Animate the number of smoothing steps, either linearly or exponentially.              | Exponential   |

*Deformation:*

| Parameter                             | Description                                                                           | Default value     |
|---------------------------------------|---------------------------------------------------------------------------------------|-------------------|
| Weight                                | Weight, interpolation or mapping scheme for deformation (see below for options).      | B-Spline Joints   |
| Inverse distance weighting exponent   | Exponent for inverse distance weighting based schemes.                                | 5                 |
| Voronoi distance                      | Influence distance (or neighborhood kernel) for Voronoi-based methods.                | 1                 |
| Gauss kernel parameter                | Epsilon parameter for a Gauss kernel used for decreasing influence with distance.     | 0.1               |
| Number of subdivisions                | Number of iterations for binary search for minimum distance computation.              | 10                |

There are six different deformation methods implemented:
- *Greedy:* Translations are defined at the feature line points, inverse distance weighting is used as interpolation method.
- *Greedy Joints:* Translations and rotations are defined at the feature line points, inverse distance weighting is used as interpolation method.
- *Voronoi:* The closest line point of the feature line is determined and only its neighbors used for interpolation using inverse distance weighting.
- *Projection:* The closest point on the feature line is determined and its position used for linear interpolation between the two adjacent line points.
- *B-Spline:* The closest point on the B-spline is determined and the translation defined as displacement between the original and the deformed B-spline.
- *B-Spline Joints:* The closest point on the B-spline is determined, and the translation and rotation defined as transformation of the original onto the deformed B-spline.

*Output:*

| Parameter                         | Description                                                                           | Default value |
|-----------------------------------|---------------------------------------------------------------------------------------|---------------|
| Output B-Spline distance          | Output array detailing the distance of the mapped point along the B-spline.           | off           |
| Output deformed grid              | Output deformed grid.                                                                 | off           |
| Output vector field               | Adjust and output vector field.                                                       | off           |
| Output resampled grid             | Resample the vector field on the original, undeformed grid.                           | off           |
| Remove elongated cells            | Remove elongated cells, which were largely deformed.                                  | off           |
| Scalar determining cell removal   | Threshold for determining which cells can be considered elongated.                    | 10            |

## Example

TODO.

## Usage

The project uses CMake as build system. To configure the project run CMake. Note that CUDA is required.

# License

This project is published under the MIT license. In the following you can find a list of contributors.

## List of contributors

- Alexander Straub, University of Stuttgart, Germany  
  (alexander.straub@visus.uni-stuttgart.de)

## MIT license

Copyright (c) 2020 University of Stuttgart Visualization Research Center

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
