  'use strict';
  const e = React.createElement;

  function Cell(props) {
    return (
      <div className="tttcell" onClick={props.onClick}>
        {props.value}
      </div>
    );
  }
  const LEN=3,SZ=LEN*LEN;
  var win=[],lose=[],nxt=Array(19683).fill().map(()=>Array(0));
  function hash(cur){
     var ret=0;
     for(let i=0;i<SZ;i++){
       ret=ret*3;
       if(cur[i]==='X') ret++;
       else if(cur[i]==='O') ret+=2;
       else if(cur[i]!==null) ret+=cur[i];
     }
     return ret;
  }
  function calc(cur,v,move){
    if(win[v]!=null) return;
    win[v]=lose[v]=1000;
    var w=calculateWinner(cur);
    if(w!=null){
    	win[v]=100; lose[v]=-100;
      return;
    }
    for(let i=0;i<9;i++) if(cur[i]===null){
    	 cur[i]=move;
    	 let x=hash(cur);
       if(win[x]==null) calc(cur,x,3-move);
       nxt[v].push([i,x]);
       win[v]=Math.min(win[v],-win[x]);
       lose[v]=Math.min(lose[v],-lose[x]);
       cur[i]=null;
    }
    if(win[v]===1000) win[v]=0;
    if(win[v]<0) win[v]++;
    if(lose[v]===1000) lose[v]=0;
  }

  function calculateWinner(board) {	
    for(let i=0;i<LEN;i++){
    	if(board[i]&&board[i]===board[i+LEN]&&board[i]===board[i+2*LEN]) return board[i];
    	if(board[i*LEN]&&board[i*LEN]===board[i*LEN+1]&&board[i*LEN]===board[i*LEN+2]) return board[i*LEN];
    }
    if(board[0]&&board[0]===board[4]&&board[0]===board[8]) return board[0];
    if(board[2]&&board[2]===board[4]&&board[2]===board[6]) return board[2];
    return null;
  }

  class TTT extends React.Component {
    constructor(props) {
      calc(Array(9).fill(null),0,1);
      super(props);
      this.state = {
        history: [
          {
            squares: Array(9).fill(null)
          }
        ],
        stepNumber: 0,
        xIsNext: true,
        difficulty:1,
        computerMove:0,
      };
    }
    move(i){
      const history = this.state.history.slice(0, this.state.stepNumber + 1);
      const current = history[history.length - 1];
      const squares = current.squares.slice();
      if (calculateWinner(squares) || squares[i]) {
        return;
      }
      squares[i] = this.state.xIsNext ? "X" : "O";
      this.setState({
        history: history.concat([
          {
            squares: squares
          }
        ]),
        stepNumber: history.length,
        xIsNext: !this.state.xIsNext,
        computerMove:0,
      });
    }
    handleClick(i) {
      if(this.state.xIsNext==this.state.computerMove) return;
      this.move(i);
      this.computerMove();
    }
    sleep(delay) {
      return new Promise(resolve => setTimeout(resolve, delay));
    }
    async computerMove(){
    	await this.sleep(500);
    	if(this.state.xIsNext!=this.state.computerMove||this.state.stepNumber==SZ) return;
      const history = this.state.history.slice(0, this.state.stepNumber + 1);
      const current = history[history.length - 1];
      const squares = current.squares.slice();
    	var x=hash(squares),moves=[];
    	if(this.state.difficulty==1){
    	  for(let i=0;i<nxt[x].length;i++)
    	    if(lose[nxt[x][i][1]]>0)
    	      moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(lose[nxt[x][i][1]]>=0)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  moves.push(nxt[x][i][0]);
      }
    	else if(this.state.difficulty==2){
    	  for(let i=0;i<nxt[x].length;i++)
    		moves.push(nxt[x][i][0]);
  	}
  	else if(this.state.difficulty==3){
    	  for(let i=0;i<nxt[x].length;i++)
    	    if(win[nxt[x][i][1]]==100)
    	      moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(win[nxt[x][i][1]]>-99)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  moves.push(nxt[x][i][0]);
  	}
  	else if(this.state.difficulty==4){
    	  for(let i=0;i<nxt[x].length;i++)
    	    if(win[nxt[x][i][1]]>=99)
    	      moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(win[nxt[x][i][1]]>=0||Math.random()<0.2&&win[nxt[x][i][1]]>-98)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(win[nxt[x][i][1]]>-98)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(win[nxt[x][i][1]]>-99)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  moves.push(nxt[x][i][0]);
  	}
  	else if(this.state.difficulty==5){
    	  for(let i=0;i<nxt[x].length;i++)
    	    if(win[nxt[x][i][1]]>0)
    	      moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  if(win[nxt[x][i][1]]>=0)
  	  	    moves.push(nxt[x][i][0]);
    	  if(!moves.length)
  	  	for(let i=0;i<nxt[x].length;i++)
  	  	  moves.push(nxt[x][i][0]);
  	}
  	console.log("Possible moves: ", moves);
  	if(moves.length) this.move(moves[Math.floor(Math.random()*moves.length)]);
    }
    toStep(step) {
      this.setState({
        stepNumber: step,
        xIsNext: step % 2 === 0
      });
    }
    restart(){
      this.toStep(0);
      this.computerMove();
    }
    undo() {
    	if(this.state.stepNumber>0) this.toStep(this.state.stepNumber-1);
    	this.setState({computerMove:-1});
    }
    redo() {
      if(this.state.stepNumber<this.state.history.length - 1) this.toStep(this.state.stepNumber+1);
    	this.setState({computerMove:-1});
    }
    setComputerMove(event){
    	this.setState({computerMove:event.target.value});
      this.computerMove();
    }
    setDifficulty(event){
    	this.setState({difficulty:event.target.value});
    }

    render() {
      const history = this.state.history;
      const current = history[this.state.stepNumber];
      const winner = calculateWinner(current.squares);

      let status;
      if (winner) {
        status = "Winner: " + winner;
      } else if(this.state.stepNumber>=SZ){
      	status="Tie";
      } else {
        status = "Next player: " + (this.state.xIsNext ? "X" : "O");
      }

    	var board=Array(SZ);
    	for(let i=0;i<SZ;i++){
    		board[i]=<Cell key={i} value={current.squares[i]} onClick={() => this.handleClick(i)}/>
    	}
      return (
        <div className="game">
            <div>{status}</div>
  	      <div className="tttboard">
  	      	{board}
  	      </div>
          <div className="game-info">
            <button onClick={() => this.restart()}>Restart</button>
            <button onClick={() => this.undo()}>Undo</button>
            <button onClick={() => this.redo()}>Redo</button>
          </div>
          <div>
            Mode:&nbsp;&nbsp;
            <select value={this.state.computerMove} onChange={(event) => this.setComputerMove(event)}>
              <option value="0">Move First</option>
              <option value="1">Move Second</option>
              <option value="-1">Two Players</option>
            </select>
            <select value={this.state.difficulty} onChange={(event) => this.setDifficulty(event)}>
              <option value="1">Trivial</option>
              <option value="2">Easy</option>
              <option value="3">Medium</option>
              <option value="4">Hard</option>
              <option value="5">Impossible</option>
            </select>
        	</div>
        </div>
      );
    }
  }

  const domContainer = document.querySelector('#tictactoecontainer');
  ReactDOM.render(e(TTT), domContainer);
  /*
    <script src="https://unpkg.com/react@17/umd/react.development.js" crossorigin></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js" crossorigin></script>
    <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
    <script src="assets/tictactoe.js" type="text/babel">
    */