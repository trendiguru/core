FROM thehipbot/web2py
COPY /local/path/to/myapp /home/www-data/web2py/applications/myapp
RUN chown -R www-data:www-data /home/www-data/web2py/applications/myapp && \
    ln -s /home/www-data/web2py/applications/myapp /home/www-data/web2py/applications/init