---
---
hour = new Date().getHours(); 
darkmode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

if darkmode and hour<21 or !darkmode and h<13
    metaTag = document.querySelector('link[href="{{'/assets/css/style.css'| relative_url}}"]')
    SSHref = metaTag.href;
    SSName = SSHref.substring(SSHref.lastIndexOf('/') + 1);
    metaTag.href = SSHref.replace(SSName, 'solarized.css');
    
