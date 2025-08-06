import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { NestExpressApplication } from '@nestjs/platform-express';
import { join } from 'path';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  
  // Enable CORS
  app.enableCors();
  
  // Serve static files from 'public' directory
  app.useStaticAssets(join(__dirname, '..', 'src', 'public'));
  
  await app.listen(3000);
}
bootstrap();
