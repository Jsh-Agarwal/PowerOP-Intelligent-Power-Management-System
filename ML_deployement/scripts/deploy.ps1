# Build production image
docker build -f Dockerfile.prod -t powerop-ml:prod .

# Tag image for registry
$VERSION = "1.0.0"
$REGISTRY = "your-registry"  # Replace with your registry
docker tag powerop-ml:prod $REGISTRY/powerop-ml:$VERSION
docker tag powerop-ml:prod $REGISTRY/powerop-ml:latest

# Push to registry
docker push $REGISTRY/powerop-ml:$VERSION
docker push $REGISTRY/powerop-ml:latest

# Deploy using docker-compose
docker-compose -f docker-compose.yml up -d
