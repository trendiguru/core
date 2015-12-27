__author__ = 'yonatan'

from subprocess import call

if __name__ == "__main__":
    # ctx = execjs.compile("""
    # function main(url){
    # var Nightmare = require('nightmare');
    # var vo = require('vo');
    #
    # vo(run)(function(err, result) {
    #   if (err) throw err;
    # });
    #
    # function *run() {
    #   var nightmare = Nightmare();
    #   var title = yield nightmare
    #     .goto(url)
    #     .evaluate(function() {
    #       return document.title;
    #     });
    #   console.log(title);
    #   yield nightmare.end();
    # }
    # }
    # """)
    # #
    # ctx.call("main", "http://cnn.com")

    call(["sudo", "xvfb-run", "--server-args='-screen 0 1024x768x24'", "node", "cnn.js"])
