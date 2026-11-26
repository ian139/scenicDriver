# Refuse stale query versions at build time, rather than silently fixing them.
FROM python:3.12.12-slim-bookworm AS checked-assets
WORKDIR /app
COPY scripts/deploy/stamp_frontend_version.py scripts/deploy/stamp_frontend_version.py
COPY apps/new_england_north/index.html apps/new_england_north/viewer.js apps/new_england_north/styles.css apps/new_england_north/
RUN python scripts/deploy/stamp_frontend_version.py --check

FROM nginx:1.28.0-alpine
COPY apps/new_england_north/nginx.beta.conf /etc/nginx/conf.d/default.conf
COPY --from=checked-assets /app/apps/new_england_north/ /usr/share/nginx/html/
EXPOSE 80
