import aiosmtplib # pyright: ignore[reportMissingImports]
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from typing import Optional

class EmailService:
    """
    Send registration confirmation emails
    """
    
    async def send_registration_confirmation(
        self,
        to_email: str,
        username: str,
        verification_code: Optional[str] = None
    ) -> bool:
        """
        Send registration confirmation email
        
        Args:
            to_email: Recipient email
            username: User's username
            verification_code: Optional verification code
        
        Returns:
            bool: Success status
        """
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = "Welcome to Face Authentication System"
            message["From"] = f"{settings.SMTP_FROM_NAME} "
            message["To"] = to_email
            
            # Email body
            text = f"""
            Hi {username},
            
            Welcome to Face Authentication System!
            
            Your registration has been completed successfully.
            
            You can now use face authentication to log in to the system.
            
            Thank you for registering!
            
            ---
            Face Authentication System
            """
            
            html = f"""
            
              
                Welcome to Face Authentication System!
                Hi {username},
                Your registration has been completed successfully.
                You can now use face authentication to log in to the system.
                
                Thank you for registering!
                
                Face Authentication System
              
            
            """
            
            # Attach parts
            part1 = MIMEText(text, "plain")
            part2 = MIMEText(html, "html")
            message.attach(part1)
            message.attach(part2)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USERNAME,
                password=settings.SMTP_PASSWORD,
                start_tls=True
            )
            
            print(f"✓ Confirmation email sent to {to_email}")
            return True
            
        except Exception as e:
            print(f"✗ Error sending email: {e}")
            return False


# Singleton instance
email_service = EmailService()