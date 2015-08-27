docker run -p 80:80 -p 443:443 -d thehipbot/web2py
Running an instance of web2py to include an existing web2py application:

docker run -p 80:80 -p 443:443 -v /local/path/to/myapp:/home/www-data/web2py/applications/myapp -d thehipbot/web2py
Example Dockerfile to build web2py application into Docker container:

FROM thehipbot/web2py
COPY /local/path/to/myapp /home/www-data/web2py/applications/myapp
RUN chown -R www-data:www-data /home/www-data/web2py/applications/myapp && \
    ln -s /home/www-data/web2py/applications/myapp /home/www-data/web2py/applications/init