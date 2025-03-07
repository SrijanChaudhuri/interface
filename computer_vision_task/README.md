sudo docker build -t sop_report_app . 
docker run -p 7860:7860 sop_report_app
check http://127.0.0.1:7860/ for ui