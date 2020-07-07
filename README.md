# alBERTi

Start a Google Cloud Deep Learning machine with PyTorch and run the next command (the hostname of the machine will be used to tag the experiment):

```bash
TAG=$(hostname) bash <(curl -s "https://raw.githubusercontent.com/linhd-postdata/alberti/master/run.sh")
```

Parameters (as envornment variables):

- `TAG`. How to tag the experiments. e.g., `alberti-roberta-base-es`. If should default to the hostname.
- `LANGS`. A comma-separated list (no spaces in between) of 2 letter language codes. Supported languages are `es` (Spanish) and `en` (English). Each language code can be prefixed with an extra `g`, which will add the raw poetry corpus extracted from Project Gutenberg 2015 DVD dump. Support for `du` (Dutch), `fr` (Frenc), and `ge` (German) is under development.
- `NFS`. Network filesystem to mount and save all the experiment runs and data to. If not given, local filesystem will be used, so be careful with the volumes termination policy.
- `NODEPS`. When set, no dependencies will be installed. This is useful for debugging.
- `NOTRAIN`. When set, no training will occur. This is useful for debugging.

A possible command in Google Cloud would look like this:
```bash
gcloud beta compute --project=$PROJECT_NANE instances create $MACHINE_NAME --zone=us-west1-b --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --metadata=startup-script=TAG=\$\(hostname\)\ bash\ \<\(curl\ -s\ \"https://raw.githubusercontent.com/linhd-postdata/alberti/master/run.sh\"\) --maintenance-policy=TERMINATE --service-account=xxxx@xxx.com --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=type=nvidia-tesla-v100,count=2 --tags=http-server,https-server --image=c2-deeplearning-pytorch-1-4-cu101-20200615 --image-project=ml-images --boot-disk-size=50GB --boot-disk-type=pd-standard --boot-disk-device-name=$MACHINE_NAME --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
```
