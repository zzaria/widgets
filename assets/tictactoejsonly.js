"use strict";const e=React.createElement;function Cell(t){return React.createElement("div",{className:"tttcell",onClick:t.onClick},t.value)}const LEN=3,SZ=LEN*LEN;var win=[],lose=[],nxt=Array(19683).fill().map(()=>Array(0));function hash(t){var e=0;for(let n=0;n<SZ;n++)e*=3,"X"===t[n]?e++:"O"===t[n]?e+=2:null!==t[n]&&(e+=t[n]);return e}function calc(t,e,n){if(null==win[e]){if(win[e]=lose[e]=1e3,null!=calculateWinner(t))return win[e]=100,void(lose[e]=-100);for(let s=0;s<9;s++)if(null===t[s]){t[s]=n;let l=hash(t);null==win[l]&&calc(t,l,3-n),nxt[e].push([s,l]),win[e]=Math.min(win[e],-win[l]),lose[e]=Math.min(lose[e],-lose[l]),t[s]=null}1e3===win[e]&&(win[e]=0),win[e]<0&&win[e]++,1e3===lose[e]&&(lose[e]=0)}}function calculateWinner(t){for(let e=0;e<LEN;e++){if(t[e]&&t[e]===t[e+LEN]&&t[e]===t[e+2*LEN])return t[e];if(t[e*LEN]&&t[e*LEN]===t[e*LEN+1]&&t[e*LEN]===t[e*LEN+2])return t[e*LEN]}return t[0]&&t[0]===t[4]&&t[0]===t[8]?t[0]:t[2]&&t[2]===t[4]&&t[2]===t[6]?t[2]:null}class TTT extends React.Component{constructor(t){calc(Array(9).fill(null),0,1),super(t),this.state={history:[{squares:Array(9).fill(null)}],stepNumber:0,xIsNext:!0,difficulty:1,computerMove:0}}move(t){const e=this.state.history.slice(0,this.state.stepNumber+1),n=e[e.length-1].squares.slice();calculateWinner(n)||n[t]||(n[t]=this.state.xIsNext?"X":"O",this.setState({history:e.concat([{squares:n}]),stepNumber:e.length,xIsNext:!this.state.xIsNext,computerMove:0}))}handleClick(t){this.state.xIsNext!=this.state.computerMove&&(this.move(t),this.computerMove())}sleep(t){return new Promise(e=>setTimeout(e,t))}async computerMove(){if(await this.sleep(500),this.state.xIsNext!=this.state.computerMove||this.state.stepNumber==SZ)return;const t=this.state.history.slice(0,this.state.stepNumber+1);var e=hash(t[t.length-1].squares.slice()),n=[];if(1==this.state.difficulty){for(let t=0;t<nxt[e].length;t++)lose[nxt[e][t][1]]>0&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)lose[nxt[e][t][1]]>=0&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)n.push(nxt[e][t][0])}else if(2==this.state.difficulty)for(let t=0;t<nxt[e].length;t++)n.push(nxt[e][t][0]);else if(3==this.state.difficulty){for(let t=0;t<nxt[e].length;t++)100==win[nxt[e][t][1]]&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>-99&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)n.push(nxt[e][t][0])}else if(4==this.state.difficulty){for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>=99&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)(win[nxt[e][t][1]]>=0||Math.random()<.2&&win[nxt[e][t][1]]>-98)&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>-98&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>-99&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)n.push(nxt[e][t][0])}else if(5==this.state.difficulty){for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>0&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)win[nxt[e][t][1]]>=0&&n.push(nxt[e][t][0]);if(!n.length)for(let t=0;t<nxt[e].length;t++)n.push(nxt[e][t][0])}console.log("Possible moves: ",n),n.length&&this.move(n[Math.floor(Math.random()*n.length)])}toStep(t){this.setState({stepNumber:t,xIsNext:t%2==0})}restart(){this.toStep(0),this.computerMove()}undo(){this.state.stepNumber>0&&this.toStep(this.state.stepNumber-1),this.setState({computerMove:-1})}redo(){this.state.stepNumber<this.state.history.length-1&&this.toStep(this.state.stepNumber+1),this.setState({computerMove:-1})}setComputerMove(t){this.setState({computerMove:t.target.value}),this.computerMove()}setDifficulty(t){this.setState({difficulty:t.target.value})}render(){const t=this.state.history[this.state.stepNumber],e=calculateWinner(t.squares);let n;n=e?"Winner: "+e:this.state.stepNumber>=SZ?"Tie":"Next player: "+(this.state.xIsNext?"X":"O");var s=Array(SZ);for(let e=0;e<SZ;e++)s[e]=React.createElement(Cell,{key:e,value:t.squares[e],onClick:()=>this.handleClick(e)});return React.createElement("div",{className:"game"},React.createElement("div",null,n),React.createElement("div",{className:"tttboard"},s),React.createElement("div",{className:"game-info"},React.createElement("button",{onClick:()=>this.restart()},"Restart"),React.createElement("button",{onClick:()=>this.undo()},"Undo"),React.createElement("button",{onClick:()=>this.redo()},"Redo")),React.createElement("div",null,"Mode:  ",React.createElement("select",{value:this.state.computerMove,onChange:t=>this.setComputerMove(t)},React.createElement("option",{value:"0"},"Move First"),React.createElement("option",{value:"1"},"Move Second"),React.createElement("option",{value:"-1"},"Two Players")),React.createElement("select",{value:this.state.difficulty,onChange:t=>this.setDifficulty(t)},React.createElement("option",{value:"1"},"Trivial"),React.createElement("option",{value:"2"},"Easy"),React.createElement("option",{value:"3"},"Medium"),React.createElement("option",{value:"4"},"Hard"),React.createElement("option",{value:"5"},"Impossible"))))}}const domContainer=document.querySelector("#tictactoecontainer");ReactDOM.render(e(TTT),domContainer);