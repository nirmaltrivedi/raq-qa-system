import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.models.document import Base
from app.core.config import settings
from app.core.logging import app_logger as logger


async def init_database():
    logger.info("Initializing database...")
    logger.info(f"Database URL: {settings.DATABASE_URL}")

    # Create async engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG
    )

    try:
        # Create all tables
        async with engine.begin() as conn:
            logger.info("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ All tables created successfully!")

        # List created tables
        async with engine.connect() as conn:
            result = await conn.execute(
                __import__('sqlalchemy').text(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                )
            )
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {', '.join(tables)}")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise
    finally:
        await engine.dispose()

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(init_database())
