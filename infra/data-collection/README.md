


## Lambda layer

```
docker run -v "$PWD":/var/task "lambci/lambda:build-python3.8" /bin/sh -c "pip install -r requirements.txt -t python/lib/python3.8/site-packages/; exit"
```

``` 
zip -r layer.zip python > /dev/null
```

```
aws lambda publish-layer-version --layer-name dataCollection --zip-file fileb://layer.zip --compatible-runtimes "python3.8"
```

```
aws lambda update-function-configuration --layers arn:aws:lambda:sa-east-1:977053370764:layer:dataCollection:1 --function-name udacity-capstone-data-collection
```