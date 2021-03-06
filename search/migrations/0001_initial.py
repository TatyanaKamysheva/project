# Generated by Django 3.1.2 on 2020-12-20 16:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PhraseVersion',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('version', models.CharField(help_text='Enter a version of phrase', max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='SearchPhrase',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('phrase', models.CharField(help_text='Enter a phrase', max_length=200)),
                ('phrase_version', models.ManyToManyField(help_text='Select a true version for this phrase', to='search.PhraseVersion')),
            ],
        ),
    ]
