---
author: aaeria
---
Abstract art using intersecting lines that divide the image into many sections.

The program first generates random lines and finds their intersections. 
Then, it converts the lines and intersections into a graph, sorting each edge in clockwise order.
Next, it starts at an unvisited node and traverses the graph, choosing the rightmost edge every time. When it arrives back at the starting location, a loop has been formed.
These loops form the sections of the image, which are then colored.

In the vector version, the image is generated in the SVG format, it can be slow when there are too many lines.
See the rasterized version for fewer customization options but better performance.

[Vector version]({{'/widgets/abstract-art.html' | relative_url}})

[Raster version]({{'/widgets/abstract-art-fast.html' | relative_url}})

## Gallery
![Example]({{'/assets/images/abstract-art-1.png' | relative_url}})
![Example]({{'/assets/images/abstract-art-2.png' | relative_url}})
![Example]({{'/assets/images/abstract-art-3.png' | relative_url}})
![Example]({{'/assets/images/abstract-art-4.png' | relative_url}})
