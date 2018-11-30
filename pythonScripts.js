
const test1 = (req, res) => {
  const spawn = require('child_process').spawn;
  const process = spawn('python', ['./scripts/test1.py', req.body.text1, req.body.text2]);

  process.stdout.on('data', data => {
    res.setHeader('Content-Type', 'application/json');
    res.send(data).end();
  });
}

const topicModelling = (req, res) => {
  const spawn = require('child_process').spawn;
  const process = spawn('python3', ['./scripts/TopicModeling.py']);
  res.setHeader('Content-Type', 'application/json');
  let accumulator = '';

  process.stdout.on('data', data => {
    const str = data.toString('utf8');
    accumulator += str;
  });

  process.on('exit', (code) => {
    console.log("Process quit with code : " + code);
    // console.log(accumulator);
    res.send(accumulator).end();
});
}

module.exports = {
  test1,
  topicModelling
}