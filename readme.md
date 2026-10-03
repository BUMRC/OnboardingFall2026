### Onboarding Project

### Running the docker container
Start the docker container by running:
```bash
docker compose up -d
```

This starts the container, and vnc so you can access it by going to `localhost:6080` the username is `ubuntu` and the password is `ubuntu`.

Enter a docker shell by running:
```
docker compose exec -u ubuntu sim bash
```


In the shell, the autonomy test case for you will be run when you run this:
```bash
cb
ros2 launch rover_autonomy autonomy.launch.py
```

#### Monte Carlo Localization
The first part of the onboarding project will be to implement monte carlo localization for the rover. All of the scripts you will be writing are in the folder `src/rover_autonomy/rover_autonomy`.

For monte carlo localization, there is a file called `mcl.py` which has the functions you need to write.


##### Some good resources will add more as I find them:
https://www.youtube.com/watch?v=MsYlueVDLI0




