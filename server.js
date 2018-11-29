const express = require('express');
const bodyParser = require("body-parser");
const pythonScripts = require('./pythonScripts');

const app = express();

app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(bodyParser.json());

app.listen(3000, () => {
  console.log('Server running on port 3000');
});

// Call a Python script supplying POST data in request body e.g. 'script/test1'
app.post('/script/:scriptname', (req, res) => {
  const scriptname = req.params.scriptname;
  pythonScripts[scriptname](req, res);
})

// Request raw data JSON type e.g. 'data/data1.json'
app.get('/data/:filename', (req, res) => {
  const n = req.params.filename;
  res.sendFile(n, {root: __dirname + '/data/' })
})

// Request a graph type e.g. 'graph/dendogram.html'
app.get('/graph/:graphname', (req, res) => {
  const n = req.params.graphname;
  res.sendFile(n, {root: __dirname + '/graphs/' })
})
