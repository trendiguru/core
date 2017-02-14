//http://msgpack.org/index.html#languages
//an attempt to recreate
//#   data = msgpack.dumps({"image": image_array_or_url})
//#    resp = requests.post(CLASSIFIER_ADDRESS, data=data, params=params)
//#    return msgpack.loads(resp.content)
//in js

var CLASSIFIER_ADDRESS = "http://13.82.136.127:8081/hydra"

var msgpack = require("msgpack-lite");

// encode from JS Object to MessagePack (Buffer)
var buffer = msgpack.encode({"foo": "bar"});

// decode from MessagePack (Buffer) to JS Object
var data = msgpack.decode(buffer); // => {"foo": "bar"}

// if encode/decode receives an invalid argument an error is thrown

