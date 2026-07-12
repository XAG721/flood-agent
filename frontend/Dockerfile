FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./frontend/
COPY 3D_visual/package*.json ./3D_visual/
RUN npm ci --prefix 3D_visual && npm ci --prefix frontend
COPY frontend ./frontend
COPY 3D_visual/public ./3D_visual/public
WORKDIR /app/frontend
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=20s --timeout=3s --retries=5 CMD wget -q -O - http://127.0.0.1/healthz || exit 1
