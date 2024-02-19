---
---
hour = new Date().getHours(); 
darkmode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

if darkmode and 13<=hour<21 or !darkmode and hour<13
    document.write('<link rel="stylesheet" type="text/css" href="{{'/assets/css/solarized.css'| relative_url}}">')
    
