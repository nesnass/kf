const express = require('express');
const bodyParser = require("body-parser");
const data1 = require('./data1.json');

const app = express();

app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(bodyParser.json());

app.listen(3000, () => {
  console.log('Server running on port 3000');
});

const pythonData = (req, res) => {
    const spawn = require('child_process').spawn;

    const process = spawn('python', ['./test.py', req.body.text1, req.body.text2]);
    process.stdout.on('data', data => {
      res.setHeader('Content-Type', 'application/json');
      res.send(data).end();
    });
}

const rawData = (req, res) => {
  res.send(data1).end();
}

app.post('/pythonData', pythonData);
app.get('/rawData', rawData);
app.get('/graph', (req, res) => {
  res.sendFile('graph.html', {root: __dirname })
})
