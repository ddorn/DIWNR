PORT ?= 8500
DEPLOY_DIR ?= /srv/camille

run:
	uv run streamlit run main.py --server.port ${PORT}

run-server:
	/root/.local/bin/uv run streamlit run main.py --server.port ${PORT}


deploy:
	git ls-files | rsync -avzP --files-from=- . pine:$(DEPLOY_DIR)
	ssh pine "cd $(DEPLOY_DIR) && make copy-service-and-restart && journalctl -u camille -f"

copy-service-and-restart:
	cp ./camille.service /etc/systemd/system/
	systemctl daemon-reload
	systemctl restart camille
